import numpy as np
import math
import cv2

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.init import normal_

from mmcv.cnn import bias_init_with_prob
from mmdet.models.builder import build_loss
from mmdet.models.utils import build_transformer
from mmdet.core import multi_apply

from mmcv.utils import Config
from models.sparse_ins import SparseInsDecoder
from .utils import inverse_sigmoid, get_contrastive_denoising_training_group
from .transformer_bricks import *
# from .aux_anchor_loss import AnchorAuxCriterion

class LATRHead(nn.Module):
    def __init__(self, args,
                 dim=128,
                 num_group=1,
                 num_convs=4,
                 in_channels=128,
                 kernel_dim=128,
                 positional_encoding=dict(
                    type='SinePositionalEncoding',
                    num_feats=128 // 2, normalize=True),
                 num_classes=21,
                 num_query=30,
                 embed_dims=128,
                 transformer=None,
                 num_reg_fcs=2,
                 depth_num=50,
                 depth_start=3,
                 top_view_region=None,
                 position_range=[-50, 3, -10, 50, 103, 10.],
                 pred_dim=10,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_reg=dict(type='L1Loss', loss_weight=2.0),
                 loss_vis=dict(type='BCEWithLogitsLoss', reduction='mean'),
                 sparse_ins_decoder=Config(
                    dict(
                        encoder=dict(
                            out_dims=64),# neck output feature channels
                        decoder=dict(
                            num_group=1,
                            output_iam=True,
                            scale_factor=1.),
                        sparse_decoder_weight=1.0,
                        )),

                 aux_criterion=Config(
                     dict(
                         cls_weight=1.0,  # neck output feature channels
                         anchor_weight=1.0,
                         matcher=dict(
                             cost_class=1,
                             cost_box=1),
                     )),
                 xs_loss_weight=1.0,
                 ys_loss_weight=0.5,
                 zs_loss_weight=5.0,
                 vis_loss_weight=1.0,
                 cls_loss_weight=20,
                 project_loss_weight=1.0,
                 trans_params=dict(
                     init_z=0, bev_h=250, bev_w=100),
                 pt_as_query=False,
                 num_pt_per_line=5,
                 num_feature_levels=1,
                 gt_project_h=20,
                 gt_project_w=30,
                 project_crit=dict(
                     type='SmoothL1Loss',
                     reduction='none'),
                 ):
        super().__init__()
        self.trans_params = dict(
            top_view_region=top_view_region,
            z_region=[position_range[2], position_range[5]])
        self.trans_params.update(trans_params)
        self.gt_project_h = gt_project_h
        self.gt_project_w = gt_project_w

        self.num_y_steps = args.num_y_steps
        self.register_buffer('anchor_y_steps',
            torch.from_numpy(args.anchor_y_steps).float())
        self.register_buffer('anchor_y_steps_dense',
            torch.from_numpy(args.anchor_y_steps_dense).float())

        project_crit['reduction'] = 'none'
        self.project_crit = getattr(
            nn, project_crit.pop('type'))(**project_crit)

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        # points num along y-axis.
        self.code_size = pred_dim
        self.num_query = num_query
        self.num_group = num_group
        self.num_pred = transformer['decoder']['num_layers']
        self.pc_range = position_range
        self.xs_loss_weight = xs_loss_weight
        self.ys_loss_weight = ys_loss_weight
        self.zs_loss_weight = zs_loss_weight
        self.vis_loss_weight = vis_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.project_loss_weight = project_loss_weight

        loss_reg['reduction'] = 'none'
        self.reg_crit = build_loss(loss_reg)
        self.cls_crit = build_loss(loss_cls)
        self.bce_loss = build_nn_loss(loss_vis)
        self.sparse_ins = SparseInsDecoder(cfg=sparse_ins_decoder)

        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.depth_start = depth_start
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.transformer = build_transformer(transformer)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # build pred layer: cls, reg, vis
        self.num_reg_fcs = num_reg_fcs
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(
            nn.Linear(
                self.embed_dims, 3))
        #        3 * self.code_size // num_pt_per_line)) # code_size = self.num_y_steps,
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        # self.aux_cls_branches = nn.ModuleList(
        #     [fc_cls for _ in range(self.num_pred)])
        # self.aux_reg_branches = nn.ModuleList(
        #     [reg_branch for _ in range(self.num_pred)])

        self.num_pt_per_line = num_pt_per_line
        self.point_embedding = nn.Embedding(
            self.num_pt_per_line, self.embed_dims)

        self.reference_points = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(True),
            nn.Linear(self.embed_dims, 3))
        # nn.Linear(self.embed_dims, 2 * self.code_size // num_pt_per_line))
        self.num_feature_levels = num_feature_levels
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))

        # self.denoising_class_embed = nn.Embedding(num_classes + 1, self.embed_dims, padding_idx=num_classes)

        # self.aux_crit = AnchorAuxCriterion(cfg=aux_criterion)
        self.dn_enabled = False
        self._init_weights()

    def _init_weights(self):
        self.transformer.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0)
        if self.cls_crit.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        normal_(self.level_embeds)

    def forward(self, input_dict, is_training=True):
        output_dict = {}
        img_feats = input_dict['x']

        ###############################################################################
        # if is_training and self.dn_enabled:
        #     denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
        #         get_contrastive_denoising_training_group(input_dict,
        #                                         2, # class
        #                                         20, # query enm
        #                                         self.denoising_class_embed,
        #                                         100, # num_denoising
        #                                         0.5, # self.label_noise_ratio,
        #                                         1.0) #elf.box_noise_scale
        # else:
        #     denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
        ###############################################################################

        if not isinstance(img_feats, (list, tuple)):
            img_feats = [img_feats]

        sparse_output = self.sparse_ins(
            img_feats[0],
            lane_idx_map=input_dict['lane_idx'],
            input_shape=input_dict['seg'].shape[-2:],
            is_training=is_training)
        # generate 2d pos emb
        B, C, H, W = img_feats[0].shape
        # B, C, H, W 4 256 90 120
        masks = img_feats[0].new_zeros((B, H, W))

        # TODO use actual mask if using padding or other aug
        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(sin_embed)

        # init query and reference pt
        query = sparse_output['inst_features'] # BxNxC
        # B, N, C -> B, N, num_anchor_per_line, C
        dd = query.unsqueeze(2)
        cc = self.point_embedding.weight
        bb = self.point_embedding.weight[None, None, ...]
        query = query.unsqueeze(2) + self.point_embedding.weight[None, None, ...] # (4, 40, 20, 256)
       
        query_embeds = self.query_embedding(query).flatten(1, 2) # (4, 800, 256)
        query = torch.zeros_like(query_embeds)
        reference_points = self.reference_points(query_embeds)
        reference_points = reference_points.sigmoid() # (4, 800, 2)
        print('LATR head reference_points.shape', reference_points.shape)
        mlvl_feats = img_feats

        feat_flatten = []
        spatial_shapes = []
        mlvl_masks = []

        assert self.num_feature_levels == len(mlvl_feats)
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2).permute(2, 0, 1) # NxBxC
            feat = feat + self.level_embeds[None, lvl:lvl+1, :].to(feat.device)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            mlvl_masks.append(torch.zeros((bs, *spatial_shape),
                                           dtype=torch.bool,
                                           device=feat.device))

        if self.transformer.with_encoder:
            mlvl_positional_encodings = []
            pos_embed2d = []
            for lvl, feat in enumerate(mlvl_feats):
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[lvl]))
                pos_embed2d.append(
                    mlvl_positional_encodings[-1].flatten(2).permute(2, 0, 1))
            pos_embed2d = torch.cat(pos_embed2d, 0)
        else:
            mlvl_positional_encodings = None
            pos_embed2d = None

        feat_flatten = torch.cat(feat_flatten, 0) # (10800, 4, 256)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=query.device) # (90, 120)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1, )),
             spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        # head
        pos_embed = None
        outs_dec, project_results, outputs_classes, outputs_coords, enc_topk_line, enc_topk_logits, dn_out_bboxes, dn_out_logits = \
            self.transformer(
                feat_flatten, None,
                query, query_embeds, pos_embed, # (4, 800, 256)
                reference_points=reference_points,
                reg_branches=self.reg_branches,
                cls_branches=self.cls_branches,
                aux_reg_branches=None,
                aux_cls_branches=None,
                img_feats=img_feats,
                lidar2img=input_dict['lidar2img'],
                pad_shape=input_dict['pad_shape'],
                sin_embed=sin_embed,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                mlvl_masks=mlvl_masks,
                mlvl_positional_encodings=mlvl_positional_encodings,
                pos_embed2d=pos_embed2d,
                image=input_dict['image'],
                # denoising_class=denoising_class,
                # denoising_bbox_unact=denoising_bbox_unact,
                # attn_mask=attn_mask,
                # dn_meta=dn_meta,
                is_training=is_training,
                **self.trans_params)
        '''
        outputs_classes 2, be cause decoder layers is 2
        outputs_coords 2
        all_cls_scores torch.Size([2, 4, 40, 21])
        all_line_preds torch.Size([2, 4, 40, 20, 1, 3])
        num_query = 40 in openlane
        '''
        # print('branch num', len(self.reg_branches))
        # print(self.reg_branches[0])
        # print('outputs_classes', len(outputs_classes))
        # print('outputs_coords', len(outputs_coords))
        all_cls_scores = torch.stack(outputs_classes)
        all_line_preds = torch.stack(outputs_coords)

        all_line_preds[..., 0] = (all_line_preds[..., 0]
            * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        print('pc_range[3]', self.pc_range[3], self.pc_range[0])
        all_line_preds[..., 1] = (all_line_preds[..., 1]
            * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        print('pc_range[4]', self.pc_range[4], self.pc_range[1])
        all_line_preds[..., 2] = (all_line_preds[..., 2]
            * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        print('pc_range[5]', self.pc_range[5], self.pc_range[2])
        print('---all_line_preds', all_line_preds.shape)
        # (4, 12, 20)
        # reshape to original format
        all_line_preds = all_line_preds.view(
            len(outputs_classes), bs, self.num_query,
            self.transformer.decoder.num_anchor_per_query, # num_anchor_per_query=num_pt_per_line,
            self.transformer.decoder.num_points_per_anchor, 3  # yz, yz
        ) # (2, 4, 12, 10, 1, 2)
        # print('all_line_preds', all_line_preds.shape)
        all_line_preds = all_line_preds.permute(0, 1, 2, 5, 3, 4)
        all_line_preds = all_line_preds.flatten(3, 5)
        
        print('+++all_line_preds', all_line_preds.shape)
        print('+++all_cls_scores', all_cls_scores.shape)

        # all_line_preds = self.process_coordinates(all_line_preds, all_cls_scores)
        #
        # print('@@@all_line_preds', all_line_preds.shape)
        # print('@@@all_cls_scores', all_cls_scores.shape)
        ###########################################################################################################
        aux_pred_logits = sparse_output['pred_logits']

        # aux_pred_logits = pred_logits.clone()
        # aux_reference_points = reference_points.clone()
        print('reference_points', reference_points.shape)
        print('###pred_logits', aux_pred_logits.shape)
        # aux_pred_logits = pred_logits.unsqueeze(0)
        # aux_reference_points = reference_points.unsqueeze(0)
        aux_reference_points = torch.zeros_like(reference_points)
        aux_reference_points[..., 0] = (reference_points[..., 0]*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        aux_reference_points[..., 1] = (reference_points[..., 1]*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        aux_reference_points[..., 2] = (reference_points[..., 2]*(self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        # (4, 12, 20)
        # reshape to original format
        print('reference_points', aux_reference_points.shape)
        aux_reference_points = aux_reference_points.view(
            bs, self.num_query,
            self.transformer.decoder.num_anchor_per_query, # num_anchor_per_query=num_pt_per_line,
            self.transformer.decoder.num_points_per_anchor, 3  # yz, yz
        ) # (2, 4, 12, 10, 1, 2)
        # print('all_line_preds', all_line_preds.shape)
        aux_reference_points = aux_reference_points.permute(0, 1, 4, 2, 3)
        aux_reference_points = aux_reference_points.flatten(2, 4)
        print('aux_reference_points', aux_reference_points.shape)
        # enc_topk_logits = torch.stack(enc_topk_logits)
        # enc_topk_line = torch.stack(enc_topk_line)
        # enc_topk_line = self.process_coordinates(enc_topk_line, enc_topk_logits)
        # print('all_cls_scores', all_cls_scores.shape)
        # print('all_line_preds', all_line_preds.shape)
        ###########################################################################################################
        # position_range [-30, -17, -5, 30, 123, 5.0]



        # print('all_line_preds', all_line_preds.shape)
        # all_line_preds torch.Size([2, 4, 40, 60]) # why use last one in evaluate
        output_dict.update({
            'all_cls_scores': all_cls_scores,
            'all_bezier_preds': all_line_preds,
            'all_aux_cls_scores': aux_pred_logits,
            'all_aux_bezier_preds': aux_reference_points,
            # 'aux_pred_logits' : enc_topk_logits,
            # 'aux_pred_bezier': enc_topk_line,
        })
        output_dict.update(sparse_output)

        if is_training:
            losses = self.get_bezier_loss(output_dict, input_dict)
            project_loss = self.get_project_loss(
                project_results, input_dict,
                h=self.gt_project_h, w=self.gt_project_w)
            losses['project_loss'] = \
                self.project_loss_weight * project_loss

            aux_loss = self.get_first_stage_loss(output_dict, input_dict)
            losses.update(aux_loss)
            # losses['aux_loss'] = self.aux_loss_weight * aux_loss

            # if denoising_class is not None:
            #     dn_out_bboxes = self.process_coordinates(dn_out_bboxes, dn_out_logits)
            #     output_dict.update({
            #         'dn_out_bboxes': dn_out_bboxes,
            #         'dn_out_logits': dn_out_logits,
            #     })
            #     losses = self.get_bezier_loss(output_dict, input_dict, dn=True)

            output_dict.update(losses)
        return output_dict

    def process_coordinates(self, output_coordinate, out_logits):

        output_coordinate[..., 0] = (output_coordinate[..., 0]
                                   * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        output_coordinate[..., 1] = (output_coordinate[..., 1]
                                    * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        output_coordinate[..., 2] = (output_coordinate[..., 2]
                                     * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        print('output_coordinate', output_coordinate.shape)
        print('len(out_logits)', len(out_logits))
        print('len(out_logits)', len(out_logits))
        print(self.num_query, self.transformer.decoder.num_anchor_per_query, self.transformer.decoder.num_points_per_anchor)
        if len(output_coordinate.shape) == 6:
            output_coordinate = output_coordinate.view(
                len(out_logits), output_coordinate.shape[1], self.num_query,
                self.transformer.decoder.num_anchor_per_query,  # num_anchor_per_query=num_pt_per_line,
                self.transformer.decoder.num_points_per_anchor, 3  # yz, yz
            )  # (2, 4, 12, 10, 1, 2)
            # print('all_line_preds', all_line_preds.shape)
            output_coordinate = output_coordinate.permute(0, 1, 2, 5, 3, 4)
            # print('all_line_preds', all_line_preds.shape)
            output_coordinate = output_coordinate.flatten(3, 5)
        else:
            output_coordinate = output_coordinate.view(
                output_coordinate.shape[0], self.num_query,
                self.transformer.decoder.num_anchor_per_query,  # num_anchor_per_query=num_pt_per_line,
                self.transformer.decoder.num_points_per_anchor, 3  # yz, yz
            )  # (2, 4, 12, 10, 1, 2)
            # print('all_line_preds', all_line_preds.shape)
            output_coordinate = output_coordinate.permute(0, 1, 4, 2, 3)
            # print('all_line_preds', all_line_preds.shape)
            output_coordinate = output_coordinate.flatten(2, 4)

        print('output_coordinate', output_coordinate.shape)
        return output_coordinate

    def get_first_stage_loss(self, aux_outputs, input_dict):
        all_cls_pred = aux_outputs['all_aux_cls_scores']
        # print('all_cls_scores', all_cls_pred)
        all_lane_pred = aux_outputs['all_aux_bezier_preds']  # torch.Size([2, 4, 40, 60])

        print('all_lane_pred', all_lane_pred.shape)  # torch.Size([2, 4, 40, 60])
        gt_lanes = input_dict['bezier_gt']
        print('gt_lanes', gt_lanes.shape)
        ground_lanes_dense = input_dict['ground_lanes_dense']
        print('ground_lanes_dense', ground_lanes_dense.shape)

        all_xs_loss = 0.0
        all_ys_loss = 0.0
        all_zs_loss = 0.0
        # all_vis_loss = 0.0
        all_cls_loss = 0.0
        matched_indices = aux_outputs['matched_indices']

        # print('matched_indices', type(matched_indices[0]))

        # glane_pred_tmp = all_lane_pred[0]  # torch.Size([4, 40, 60])
        # glane_pred = glane_pred_tmp.view(  # torch.Size([4, 1, 40, 60])
        #     glane_pred_tmp.shape[0],
        #     self.num_group,
        #     self.num_query,
        #     glane_pred_tmp.shape[-1])
        # lpred = glane_pred[:, 0, ...]

        # print('lpred', lpred.shape) # torch.Size([4, 40, 60])
        # print('grup and glane_pred_tmp', self.num_group, len(matched_indices[0]), glane_pred.shape)


        gcls_pred = all_cls_pred
        glane_pred = all_lane_pred
        # print('+++++++', 'layer_idx', layer_idx)
        # print('+++++++', 'all_lane_pred', all_lane_pred.shape)
        # print('+++++++', 'glane_pred', glane_pred.shape)
        glane_pred = glane_pred.view(
            glane_pred.shape[0],
            self.num_group, # 1
            self.num_query, # 12
            glane_pred.shape[-1])
        # print('$$$$$', 'glane_pred', glane_pred.shape)
        gcls_pred = gcls_pred.view(
            gcls_pred.shape[0],
            self.num_group,
            self.num_query,
            gcls_pred.shape[-1])

        per_xs_loss = 0.0
        per_ys_loss = 0.0
        per_zs_loss = 0.0
        # per_vis_loss = 0.0
        per_cls_loss = 0.0
        batch_size = len(matched_indices[0])  # list batch_size [[[3, 27], [1,0]], [],[], [], [0, 2, 9, 16, 27, 36]
        # print('XXX layer_idx', batch_size)
        print('matched_indices', matched_indices)
        for b_idx in range(len(matched_indices[0])):  # indices means query index
            for group_idx in range(self.num_group):
                pred_idx = matched_indices[group_idx][b_idx][0]
                gt_idx = matched_indices[group_idx][b_idx][1]

                cls_pred = gcls_pred[:, group_idx, ...]
                lane_pred = glane_pred[:, group_idx, ...]  # torch.Size([4, 40, 60])

                if gt_idx.shape[0] < 1:
                    cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                    cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)
                    per_cls_loss = per_cls_loss + cls_loss
                    per_xs_loss = per_xs_loss + 0.0 * lane_pred[b_idx].mean()
                    continue

                pos_lane_pred = lane_pred[b_idx][pred_idx]  # orch.Size([2, 60])
                # print('ITER', 'pos_lane_pred', pos_lane_pred.shape)
                # exit(0)
                # print('b_idx', b_idx, 'gt_idx', gt_idx)
                gt_lane = gt_lanes[b_idx][gt_idx]

                pred_xs = pos_lane_pred[:, :5]
                pred_ys = pos_lane_pred[:, 5:10]
                pred_zs = pos_lane_pred[:, 10:15]
                # pred_vis = pos_lane_pred[:, 2 * self.code_size:]
                gt_xs = gt_lane[:, :5]
                gt_ys = gt_lane[:, 5:10]
                gt_zs = gt_lane[:, 10:15]
                # gt_vis = gt_lane[:, 2 * self.code_size:3 * self.code_size]

                # loc_mask = gt_vis > 0
                # aaa = torch.clamp(loc_mask.sum(), 1)
                # print('aaa', aaa.shape, aaa, 'loc_mask.sum()', loc_mask.sum())
                xs_loss = self.reg_crit(pred_xs, gt_xs)
                ys_loss = self.reg_crit(pred_ys, gt_ys)
                zs_loss = self.reg_crit(pred_zs, gt_zs)
                # print('xs_loss###########################################', xs_loss.shape)
                xs_loss = xs_loss.sum() / 5
                ys_loss = ys_loss.sum() / 5
                zs_loss = zs_loss.sum() / 5

                # vis_loss = self.bce_loss(pred_vis, gt_vis)

                cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                cls_target[pred_idx] = torch.argmax(
                    gt_lane[:, 15:], dim=1)
                # print('gcls_pred.shape', gcls_pred.shape)
                # print('-----------------------------------classfication loss, each batch---------------------------------')
                # print('prediction class', cls_pred[b_idx].shape)
                # print('target class', cls_target.shape)
                # print('prediction result', cls_pred[b_idx])
                # print('target result', cls_target)
                cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)

                per_xs_loss += xs_loss
                per_ys_loss += ys_loss
                per_zs_loss += zs_loss
                # per_vis_loss += vis_loss
                per_cls_loss += cls_loss

                    # print('per_xs_loss###########################################', per_xs_loss.shape)
        # print('all_xs_loss, num', len(all_xs_loss), 'shape', all_xs_loss[0].shape)
        # exit(0)
        all_xs_loss = all_xs_loss
        all_zs_loss = all_zs_loss
        all_ys_loss = all_ys_loss
        # all_vis_loss = sum(all_vis_loss)
        all_cls_loss = all_cls_loss


        return dict(
            all_aux_xs_loss=self.xs_loss_weight * all_xs_loss,
            all_aux_ys_loss=self.ys_loss_weight * all_ys_loss,
            all_aux_zs_loss=self.zs_loss_weight * all_zs_loss,
            # all_vis_loss=self.vis_loss_weight * all_vis_loss,
            all_aux_cls_loss=self.cls_loss_weight * all_cls_loss,
        )

    def get_aux_loss(self, aux_outputs, input_dict):

        losses = self.aux_crit(outputs=dict(
                        aux_pred_logits=aux_outputs['aux_pred_logits'],
                        aux_pred_bezier=aux_outputs['aux_pred_bezier'],
                    ),
                    targets=dict(
                        lane_labels=input_dict['lane_labels'],
                        beizer_gt=input_dict['lane_labels'],
                        lane_num=input_dict['lane_num']
                    )
        )

        for k in losses.keys():
                losses[k] *= self.aux_loss_weight

        return losses

    def get_project_loss(self, results, input_dict, h=20, w=30):
        gt_lane = input_dict['ground_lanes_dense']
        gt_ys = self.anchor_y_steps_dense.clone()
        code_size = gt_ys.shape[0]
        gt_xs = gt_lane[..., :code_size]
        gt_zs = gt_lane[..., code_size : 2*code_size]
        gt_vis = gt_lane[..., 2*code_size:3*code_size]
        gt_ys = gt_ys[None, None, :].expand_as(gt_xs)
        gt_points = torch.stack([gt_xs, gt_ys, gt_zs], dim=-1)

        B = results[0].shape[0]
        ref_3d_home = F.pad(gt_points, (0, 1), value=1)
        coords_img = ground2img(
            ref_3d_home,
            h, w,
            input_dict['lidar2img'],
            input_dict['pad_shape'], mask=gt_vis)

        all_loss = 0.
        for projct_result in results:
            projct_result = F.interpolate(
                projct_result,
                size=(h, w),
                mode='nearest')
            gt_proj = coords_img.clone()

            mask = (gt_proj[:, -1, ...] > 0) * (projct_result[:, -1, ...] > 0)
            diff_loss = self.project_crit(
                projct_result[:, :3, ...],
                gt_proj[:, :3, ...],
            )
            diff_y_loss = diff_loss[:, 1, ...]
            diff_z_loss = diff_loss[:, 2, ...]
            diff_loss = diff_y_loss * 0.1 + diff_z_loss
            diff_loss = (diff_loss * mask).sum() / torch.clamp(mask.sum(), 1)
            all_loss = all_loss + diff_loss

        return all_loss / len(results)

    def get_bezier_loss(self, output_dict, input_dict, aux=False):
        '''
        ground_lanes[i][0: self.num_y_steps] = xs
        ground_lanes[i][self.num_y_steps:2*self.num_y_steps] = zs
        ground_lanes[i][2*self.num_y_steps:3*self.num_y_steps] = vis * 1.0
        ground_lanes[i][self.anchor_dim - self.num_category] = 0.0
        ground_lanes[i][self.anchor_dim - self.num_category + _gt_laneline_category_org[i]] = 1.0 #the i-th lane
        '''
        paths = input_dict['lane_num']
        print(paths.shape)
        print(paths)
        print('GT')
        '''
        gt_lanes torch.Size([4, 20, 81]) # 3(x, y, visible) * self.num_y_steps + args.num_category
        ground_lanes_dense torch.Size([4, 20, 600])
        '''
        if aux is True:
            all_cls_pred = output_dict['all_aux_cls_scores']
            # print('all_cls_scores', all_cls_pred)
            all_lane_pred = output_dict['all_aux_bezier_preds']  # torch.Size([2, 4, 40, 60])
        else:
            all_cls_pred = output_dict['all_cls_scores']
            # print('all_cls_scores', all_cls_pred)
            all_lane_pred = output_dict['all_bezier_preds']  # torch.Size([2, 4, 40, 60])
        # print('all_lane_pred', all_lane_pred.shape)  # torch.Size([2, 4, 40, 60])
        gt_lanes = input_dict['bezier_gt']
        # print('gt_lanes', gt_lanes.shape)
        ground_lanes_dense = input_dict['ground_lanes_dense']
        # print('ground_lanes_dense', ground_lanes_dense.shape)

        all_xs_loss = 0.0
        all_ys_loss = 0.0
        all_zs_loss = 0.0
        # all_vis_loss = 0.0
        all_cls_loss = 0.0
        matched_indices = output_dict['matched_indices']
        num_layers = all_lane_pred.shape[0]
        # print('matched_indices', type(matched_indices[0]))

        # glane_pred_tmp = all_lane_pred[0]  # torch.Size([4, 40, 60])
        # glane_pred = glane_pred_tmp.view(  # torch.Size([4, 1, 40, 60])
        #     glane_pred_tmp.shape[0],
        #     self.num_group,
        #     self.num_query,
        #     glane_pred_tmp.shape[-1])
        # lpred = glane_pred[:, 0, ...]

        # print('lpred', lpred.shape) # torch.Size([4, 40, 60])
        # print('grup and glane_pred_tmp', self.num_group, len(matched_indices[0]), glane_pred.shape)

        def single_layer_loss(layer_idx):
            gcls_pred = all_cls_pred[layer_idx]
            glane_pred = all_lane_pred[layer_idx]  # torch.Size([2, 40, 60])
            # print('+++++++', 'layer_idx', layer_idx)
            # print('+++++++', 'all_lane_pred', all_lane_pred.shape)
            # print('+++++++', 'glane_pred', glane_pred.shape)
            glane_pred = glane_pred.view(
                glane_pred.shape[0],
                self.num_group, # 1
                self.num_query, # 12
                glane_pred.shape[-1])
            # print('$$$$$', 'glane_pred', glane_pred.shape)
            gcls_pred = gcls_pred.view(
                gcls_pred.shape[0],
                self.num_group,
                self.num_query,
                gcls_pred.shape[-1])

            per_xs_loss = 0.0
            per_ys_loss = 0.0
            per_zs_loss = 0.0
            # per_vis_loss = 0.0
            per_cls_loss = 0.0
            batch_size = len(matched_indices[0])  # list batch_size [[[3, 27], [1,0]], [],[], [], [0, 2, 9, 16, 27, 36]
            # print('XXX layer_idx', batch_size)
            print('matched_indices', matched_indices)
            for b_idx in range(len(matched_indices[0])):  # indices means query index
                for group_idx in range(self.num_group):
                    pred_idx = matched_indices[group_idx][b_idx][0]
                    gt_idx = matched_indices[group_idx][b_idx][1]

                    cls_pred = gcls_pred[:, group_idx, ...]
                    lane_pred = glane_pred[:, group_idx, ...]  # torch.Size([4, 40, 60])

                    if gt_idx.shape[0] < 1:
                        cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                        cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)
                        per_cls_loss = per_cls_loss + cls_loss
                        per_xs_loss = per_xs_loss + 0.0 * lane_pred[b_idx].mean()
                        continue

                    pos_lane_pred = lane_pred[b_idx][pred_idx]  # orch.Size([2, 60])
                    # print('ITER', 'pos_lane_pred', pos_lane_pred.shape)
                    # exit(0)
                    # print('b_idx', b_idx, 'gt_idx', gt_idx)
                    gt_lane = gt_lanes[b_idx][gt_idx]

                    pred_xs = pos_lane_pred[:, :5]
                    pred_ys = pos_lane_pred[:, 5:10]
                    pred_zs = pos_lane_pred[:, 10:15]
                    # pred_vis = pos_lane_pred[:, 2 * self.code_size:]
                    gt_xs = gt_lane[:, :5]
                    gt_ys = gt_lane[:, 5:10]
                    gt_zs = gt_lane[:, 10:15]
                    # gt_vis = gt_lane[:, 2 * self.code_size:3 * self.code_size]

                    # loc_mask = gt_vis > 0
                    # aaa = torch.clamp(loc_mask.sum(), 1)
                    # print('aaa', aaa.shape, aaa, 'loc_mask.sum()', loc_mask.sum())
                    xs_loss = self.reg_crit(pred_xs, gt_xs)
                    ys_loss = self.reg_crit(pred_ys, gt_ys)
                    zs_loss = self.reg_crit(pred_zs, gt_zs)
                    # print('xs_loss###########################################', xs_loss.shape)
                    xs_loss = xs_loss.sum() / 5
                    ys_loss = ys_loss.sum() / 5
                    zs_loss = zs_loss.sum() / 5

                    # vis_loss = self.bce_loss(pred_vis, gt_vis)

                    cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                    cls_target[pred_idx] = torch.argmax(
                        gt_lane[:, 15:], dim=1)
                    # print('gcls_pred.shape', gcls_pred.shape)
                    # print('-----------------------------------classfication loss, each batch---------------------------------')
                    # print('prediction class', cls_pred[b_idx].shape)
                    # print('target class', cls_target.shape)
                    # print('prediction result', cls_pred[b_idx])
                    # print('target result', cls_target)
                    cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)

                    per_xs_loss += xs_loss
                    per_ys_loss += ys_loss
                    per_zs_loss += zs_loss
                    # per_vis_loss += vis_loss
                    per_cls_loss += cls_loss

                    # print('per_xs_loss###########################################', per_xs_loss.shape)

            return tuple(map(lambda x: x / batch_size / self.num_group,
                             [per_xs_loss, per_zs_loss, per_ys_loss, per_cls_loss]))

        all_xs_loss, all_zs_loss, all_ys_loss, all_cls_loss = multi_apply(
            single_layer_loss, range(all_lane_pred.shape[0]))  # what does it means? [0, 1]
        # print('all_xs_loss, num', len(all_xs_loss), 'shape', all_xs_loss[0].shape)
        # exit(0)
        all_xs_loss = sum(all_xs_loss) / num_layers
        all_zs_loss = sum(all_zs_loss) / num_layers
        all_ys_loss = sum(all_ys_loss) / num_layers
        # all_vis_loss = sum(all_vis_loss) / num_layers
        all_cls_loss = sum(all_cls_loss) / num_layers

        if aux == False:
            return dict(
                all_xs_loss=self.xs_loss_weight * all_xs_loss,
                all_ys_loss=self.ys_loss_weight * all_ys_loss,
                all_zs_loss=self.zs_loss_weight * all_zs_loss,
                # all_vis_loss=self.vis_loss_weight * all_vis_loss,
                all_cls_loss=self.cls_loss_weight * all_cls_loss,
            )
        else:
            return dict(
                all_aux_xs_loss=self.xs_loss_weight * all_xs_loss,
                all_aux_ys_loss=self.ys_loss_weight * all_ys_loss,
                all_aux_zs_loss=self.zs_loss_weight * all_zs_loss,
                # all_vis_loss=self.vis_loss_weight * all_vis_loss,
                all_aux_cls_loss=self.cls_loss_weight * all_cls_loss,
            )

    def get_loss(self, output_dict, input_dict):
        '''
        ground_lanes[i][0: self.num_y_steps] = xs
        ground_lanes[i][self.num_y_steps:2*self.num_y_steps] = zs
        ground_lanes[i][2*self.num_y_steps:3*self.num_y_steps] = vis * 1.0
        ground_lanes[i][self.anchor_dim - self.num_category] = 0.0
        ground_lanes[i][self.anchor_dim - self.num_category + _gt_laneline_category_org[i]] = 1.0 #the i-th lane
        '''
        paths = input_dict['lane_num']
        print(paths.shape)
        print(paths)
        print('GT')
        '''
        gt_lanes torch.Size([4, 20, 81]) # 3(x, y, visible) * self.num_y_steps + args.num_category
        ground_lanes_dense torch.Size([4, 20, 600])
        '''
        all_cls_pred = output_dict['all_cls_scores']
        # print('all_cls_scores', all_cls_pred)
        all_lane_pred = output_dict['all_line_preds'] # torch.Size([2, 4, 40, 60])
        print('all_lane_pred', all_lane_pred.shape) # torch.Size([2, 4, 40, 60])
        gt_lanes = input_dict['ground_lanes']
        print('gt_lanes', gt_lanes.shape)
        ground_lanes_dense = input_dict['ground_lanes_dense']
        print('ground_lanes_dense', ground_lanes_dense.shape)


        all_xs_loss = 0.0
        all_zs_loss = 0.0
        all_vis_loss = 0.0
        all_cls_loss = 0.0
        matched_indices = output_dict['matched_indices']
        num_layers = all_lane_pred.shape[0]
        print('matched_indices', type(matched_indices[0]))

        glane_pred_tmp = all_lane_pred[0] # torch.Size([4, 40, 60])
        glane_pred = glane_pred_tmp.view( # torch.Size([4, 1, 40, 60])
            glane_pred_tmp.shape[0],
            self.num_group,
            self.num_query,
            glane_pred_tmp.shape[-1])
        lpred = glane_pred[:, 0, ...]
        # print('lpred', lpred.shape) # torch.Size([4, 40, 60])
        # print('grup and glane_pred_tmp', self.num_group, len(matched_indices[0]), glane_pred.shape)

        def single_layer_loss(layer_idx):
            gcls_pred = all_cls_pred[layer_idx]
            glane_pred = all_lane_pred[layer_idx] # torch.Size([2, 40, 60])
            # print('+++++++', 'layer_idx', layer_idx)
            # print('+++++++', 'all_lane_pred', all_lane_pred.shape)
            # print('+++++++', 'glane_pred', glane_pred.shape)
            glane_pred = glane_pred.view(
                glane_pred.shape[0],
                self.num_group,
                self.num_query,
                glane_pred.shape[-1])
            # print('$$$$$', 'glane_pred', glane_pred.shape)
            gcls_pred = gcls_pred.view(
                gcls_pred.shape[0],
                self.num_group,
                self.num_query,
                gcls_pred.shape[-1])

            per_xs_loss = 0.0
            per_zs_loss = 0.0
            per_vis_loss = 0.0
            per_cls_loss = 0.0
            batch_size = len(matched_indices[0]) #list batch_size [[[3, 27], [1,0]], [],[], [], [0, 2, 9, 16, 27, 36]
            # print('XXX layer_idx', batch_size)

            for b_idx in range(len(matched_indices[0])): # indices means query index
                for group_idx in range(self.num_group):
                    pred_idx = matched_indices[group_idx][b_idx][0]
                    gt_idx = matched_indices[group_idx][b_idx][1]

                    cls_pred = gcls_pred[:, group_idx, ...]
                    lane_pred = glane_pred[:, group_idx, ...] # torch.Size([4, 40, 60])

                    if gt_idx.shape[0] < 1:
                        cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                        cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)
                        per_cls_loss = per_cls_loss + cls_loss
                        per_xs_loss = per_xs_loss + 0.0 * lane_pred[b_idx].mean()
                        continue

                    pos_lane_pred = lane_pred[b_idx][pred_idx] # orch.Size([2, 60])
                    # print('ITER', 'pos_lane_pred', pos_lane_pred.shape)
                    # exit(0)
                    # print('b_idx', b_idx, 'gt_idx', gt_idx)
                    gt_lane = gt_lanes[b_idx][gt_idx]

                    pred_xs = pos_lane_pred[:, :self.code_size]
                    pred_zs = pos_lane_pred[:, self.code_size : 2*self.code_size]
                    pred_vis = pos_lane_pred[:, 2*self.code_size:]
                    gt_xs = gt_lane[:, :self.code_size]
                    gt_zs = gt_lane[:, self.code_size : 2*self.code_size]
                    gt_vis = gt_lane[:, 2*self.code_size:3*self.code_size]


                    loc_mask = gt_vis > 0
                    aaa = torch.clamp(loc_mask.sum(), 1)
                    # print('aaa', aaa.shape, aaa, 'loc_mask.sum()', loc_mask.sum())
                    xs_loss = self.reg_crit(pred_xs, gt_xs)
                    zs_loss = self.reg_crit(pred_zs, gt_zs)
                    # print('xs_loss###########################################', xs_loss.shape)
                    xs_loss = (xs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), 1)
                    # print('xs_loss###########################################', xs_loss.shape)
                    zs_loss = (zs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), 1)
                    vis_loss = self.bce_loss(pred_vis, gt_vis)

                    cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                    cls_target[pred_idx] = torch.argmax(
                        gt_lane[:, 3*self.code_size:], dim=1)
                    cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)

                    per_xs_loss += xs_loss
                    per_zs_loss += zs_loss
                    per_vis_loss += vis_loss
                    per_cls_loss += cls_loss

                    # print('per_xs_loss###########################################', per_xs_loss.shape)

            return tuple(map(lambda x: x / batch_size / self.num_group,
                             [per_xs_loss, per_zs_loss, per_vis_loss, per_cls_loss]))

        all_xs_loss, all_zs_loss, all_vis_loss, all_cls_loss = multi_apply(
            single_layer_loss, range(all_lane_pred.shape[0])) # what does it means? [0, 1]
        # print('all_xs_loss, num', len(all_xs_loss), 'shape', all_xs_loss[0].shape)
        # exit(0)
        all_xs_loss = sum(all_xs_loss) / num_layers
        all_zs_loss = sum(all_zs_loss) / num_layers
        all_vis_loss = sum(all_vis_loss) / num_layers
        all_cls_loss = sum(all_cls_loss) / num_layers

        return dict(
            all_xs_loss=self.xs_loss_weight * all_xs_loss,
            all_zs_loss=self.zs_loss_weight * all_zs_loss,
            all_vis_loss=self.vis_loss_weight * all_vis_loss,
            all_cls_loss=self.cls_loss_weight * all_cls_loss,
        )

    @staticmethod
    def get_reference_points(H, W, bs=1, device='cuda', dtype=torch.float):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1) 
        return ref_2d

def build_nn_loss(loss_cfg):
    crit_t = loss_cfg.pop('type')
    return getattr(nn, crit_t)(**loss_cfg)
