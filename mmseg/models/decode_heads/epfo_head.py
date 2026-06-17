# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS  # ⚠改动：旧版注册器
from .cascade_decode_head import BaseCascadeDecodeHead
from ..losses import accuracy  # ⚠改动：路径变化
from mmcv.ops import point_sample


def calculate_uncertainty(seg_logits):
    top2_scores = torch.topk(seg_logits, k=2, dim=1)[0]
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)


def one_hot(label, n_classes, requires_grad=False, ignore_index=255):
    device = label.device
    zeros = label.new_zeros(label.shape).to(device)
    label_zeros = torch.where(((label >= 0) & (label == ignore_index)), zeros, label)

    one_hot_label = torch.eye(n_classes, device=device, requires_grad=requires_grad)[label_zeros]
    one_hot_label[:, :, :, 0] = torch.where(((label >= 0) & (label == ignore_index)), zeros,
                                            one_hot_label[:, :, :, 0].type(torch.LongTensor).to(device))
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)
    return one_hot_label


@HEADS.register_module()
class EPFO_Head(BaseCascadeDecodeHead):
    def __init__(self,
                 num_fcs=3,
                 theta=3,
                 coarse_pred_each_layer=True,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=False),
                 **kwargs):
        super(EPFO_Head, self).__init__(
            input_transform='multiple_select',
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        if point_sample is None:
            raise RuntimeError('Please install mmcv-full for point_sample ops')

        self.num_fcs = num_fcs
        self.theta = theta
        self.coarse_pred_each_layer = coarse_pred_each_layer

        fc_in_channels = sum(self.in_channels) + self.num_classes
        fc_channels = self.channels
        self.fcs = nn.ModuleList()
        for k in range(num_fcs):
            fc = ConvModule(
                fc_in_channels,
                fc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += self.num_classes if self.coarse_pred_each_layer else 0

        self.fc_seg = nn.Conv1d(
            fc_in_channels,
            self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

        if hasattr(self, 'conv_seg'):
            delattr(self, 'conv_seg')  # ⚠旧版兼容删除原有 conv_seg

    def cls_seg(self, feat):
        if hasattr(self, 'dropout') and self.dropout is not None:
            feat = self.dropout(feat)
        output = self.fc_seg(feat)
        return output

    def forward(self, fine_grained_point_feats, coarse_point_feats):
        x = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
        for fc in self.fcs:
            x = fc(x)
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_point_feats), dim=1)
        return self.cls_seg(x)

    def _get_fine_grained_point_feats(self, x, points):
        fine_grained_feats_list = [
            point_sample(x, points, align_corners=self.align_corners)
        ]
        if len(fine_grained_feats_list) > 1:
            fine_grained_feats = torch.cat(fine_grained_feats_list, dim=1)
        else:
            fine_grained_feats = fine_grained_feats_list[0]
        return fine_grained_feats

    def _get_coarse_point_feats(self, prev_output, points):
        coarse_feats = point_sample(prev_output, points, align_corners=self.align_corners)
        return coarse_feats

    def _get_boundry(self, pred):
        n, c, _, _ = pred.shape
        coarse_label = F.softmax(pred, dim=1).argmax(dim=1)
        one_hot_gt = one_hot(coarse_label, c, ignore_index=255)
        pred_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b -= 1 - one_hot_gt
        pred_b = torch.sum(pred_b, dim=1)
        pred_b[pred_b > 0] = 1.0
        return pred_b

    # ⚠ 旧版 loss 接口：inputs, prev_output, gt_semantic_seg, train_cfg
    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg):
        x = self._transform_inputs(inputs)

        coarse_label = resize(
            input=prev_output,
            size=x[0].shape[-2:],  # ⚠旧API用 tuple size
            mode='bilinear',
            align_corners=self.align_corners)
        fine_label = resize(
            input=x[0],
            size=x[0].shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)

        with torch.no_grad():
            boundry = self._get_boundry(coarse_label)
            points = self.get_points_train(coarse_label, boundry, calculate_uncertainty, cfg=train_cfg)

        fine_grained_point_feats = self._get_fine_grained_point_feats(fine_label, points)
        coarse_point_feats = self._get_coarse_point_feats(coarse_label, points)
        point_logits = self.forward(fine_grained_point_feats, coarse_point_feats)

        losses = self.loss_by_feat(point_logits, points, gt_semantic_seg)
        return losses

    # ⚠ predict接口调整为 forward_test
    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        x = self._transform_inputs(inputs)
        coarse_label = resize(
            input=prev_output,
            size=x[0].shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        MSCA_out = coarse_label.clone()
        fine_label = resize(
            input=x[0],
            size=x[0].shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        batch_size, channels, height, width = coarse_label.shape
        boundry = self._get_boundry(coarse_label)
        point_indices, points = self.get_points_test(coarse_label, boundry, calculate_uncertainty, cfg=test_cfg)

        fine_grained_point_feats = self._get_fine_grained_point_feats(fine_label, points)
        coarse_point_feats = self._get_coarse_point_feats(coarse_label, points)
        point_logits = self.forward(fine_grained_point_feats, coarse_point_feats)

        point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
        coarse_label = coarse_label.reshape(batch_size, channels, height * width)
        coarse_label = coarse_label.scatter_(2, point_indices, point_logits)
        coarse_label = coarse_label.view(batch_size, channels, height, width)

        return self.predict_by_feat(MSCA_out, boundry, points, coarse_label, img_metas)

    def predict_by_feat(self, seg_logits, boundry, points, coarse_label, img_metas):
        seg_logits = resize(
            input=seg_logits,
            size=img_metas[0]['img_shape'][:2],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits

    def loss_by_feat(self, point_logits, points, gt_semantic_seg):
        point_label = point_sample(
            gt_semantic_seg.float(),
            points,
            mode='nearest',
            align_corners=self.align_corners)
        point_label = point_label.squeeze(1).long()

        loss = dict()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_module in losses_decode:
            loss['point_' + loss_module.loss_name] = loss_module(
                point_logits, point_label, ignore_index=self.ignore_index)
        loss['acc_point'] = accuracy(point_logits, point_label, ignore_index=self.ignore_index)
        return loss
    
    def get_points_train(self, seg_logits, boundry, uncertainty_func, cfg):
        boundry_coords = [(torch.nonzero(boundry[i] > 0, as_tuple=False) / (self.channels - 1)).shape[0] for i in range(boundry.shape[0])]
        num_points = int(min(boundry_coords)*cfg.ratio)
        if num_points == 0:
            num_points = int(max(boundry_coords)*cfg.ratio)
        if num_points == 0:
            num_points = 1
        uncertainty_map = uncertainty_func(seg_logits)

        batch_size, _, height, width = uncertainty_map.shape
        h_step = 1.0 / height
        w_step = 1.0 / width
        assert height * width >= num_points
        point_coordss = []
        uncertainty_map_zeros=uncertainty_map * boundry.unsqueeze(1)
        list_zeros=[(torch.sum(uncertainty_map_zeros[i] != 0)) for i in range(uncertainty_map_zeros.shape[0])]


        for i in range(len(list_zeros)):
            if list_zeros[i] >= num_points:
                uncertainty_map0 = uncertainty_map[i:i+1,:,:,:].clone()
                uncertainty_map0[uncertainty_map_zeros[i:i+1,:,:,:] == 0]=torch.min(uncertainty_map0)-1.0
                uncertainty_map0 = uncertainty_map0.view(1, height * width)
                point_indices = uncertainty_map0.topk(num_points, dim=1)[1]

                point_coords = torch.zeros(
                    1,
                    num_points,
                    2,
                    dtype=torch.float,
                    device=seg_logits.device)
                point_coords[:, :, 0] = (point_indices % width).float() * w_step
                point_coords[:, :, 1] = (point_indices // width).float() * h_step

            else:
                uncertainty_map1=uncertainty_map[i:i+1,:,:,:].clone()
                uncertainty_map1[uncertainty_map_zeros[i:i+1,:,:,:] == 0]=torch.min(uncertainty_map1)-1.0
                uncertainty_map1 = uncertainty_map1.view(1, height * width)
                point_indices0 = uncertainty_map1.topk(list_zeros[i], dim=1)[1]
                point_coords = torch.zeros(
                    1,
                    num_points,
                    2,
                    dtype=torch.float,
                    device=seg_logits.device)
                point_coords[:, :list_zeros[i], 0] = (point_indices0 % width).float() * w_step
                point_coords[:, :list_zeros[i], 1] = (point_indices0 // width).float() * h_step

                uncertainty_map1 = uncertainty_map[i:i+1,:,:,:].clone()
                uncertainty_map_zeros = uncertainty_map[i:i+1,:,:,:] * (1-boundry).unsqueeze(1)
                uncertainty_map1[uncertainty_map_zeros[i:i+1,:,:,:] == 0]=torch.min(uncertainty_map1)-1.0
                uncertainty_map1 = uncertainty_map1.view(1, height * width)

                point_indices1 = uncertainty_map1.topk(num_points-list_zeros[i], dim=1)[1]

                point_coords[:, list_zeros[i]:, 0] = (point_indices1 % width).float() * w_step
                point_coords[:, list_zeros[i]:, 1] = (point_indices1 // width).float() * h_step

            point_coordss.append(point_coords)

        point_coords = torch.cat(point_coordss, dim=0)

        return point_coords

    def get_points_test(self, seg_logits, boundry, uncertainty_func, cfg):
        boundry_coords = [(torch.nonzero(boundry[i] > 0, as_tuple=False) / (self.channels - 1)).shape[0] for i in range(boundry.shape[0])]
        num_points = int(min(boundry_coords)*cfg.ratio)
        if num_points == 0:
            num_points = int(max(boundry_coords)*cfg.ratio)
        if num_points == 0:
            num_points = 1
        # print(num_points)
        uncertainty_map = uncertainty_func(seg_logits)

        batch_size, _, height, width = uncertainty_map.shape
        h_step = 1.0 / height
        w_step = 1.0 / width
        assert height * width >= num_points
        point_coordss = []
        point_indicess = []
        uncertainty_map_zeros=uncertainty_map * boundry.unsqueeze(1)
        list_zeros=[(torch.sum(uncertainty_map_zeros[i] != 0)) for i in range(uncertainty_map_zeros.shape[0])]

        for i in range(len(list_zeros)):
            if list_zeros[i] >= num_points:
                uncertainty_map0 = uncertainty_map[i:i+1,:,:,:].clone()
                uncertainty_map0[uncertainty_map_zeros[i:i+1,:,:,:] == 0]=torch.min(uncertainty_map0)-1.0

                uncertainty_map0 = uncertainty_map0.view(1, height * width)
                point_indices = uncertainty_map0.topk(num_points, dim=1)[1]

                point_coords = torch.zeros(
                    1,
                    num_points,
                    2,
                    dtype=torch.float,
                    device=seg_logits.device)
                point_coords[:, :, 0] = (point_indices % width).float() * w_step
                point_coords[:, :, 1] = (point_indices // width).float() * h_step

            else:
                uncertainty_map1=uncertainty_map[i:i+1,:,:,:].clone()
                uncertainty_map1[uncertainty_map_zeros[i:i+1,:,:,:] == 0]=torch.min(uncertainty_map1)-1.0
                uncertainty_map1 = uncertainty_map1.view(1, height * width)
                point_indices0 = uncertainty_map1.topk(list_zeros[i], dim=1)[1]
                point_coords = torch.zeros(
                    1,
                    num_points,
                    2,
                    dtype=torch.float,
                    device=seg_logits.device)
                point_coords[:, :list_zeros[i], 0] = (point_indices0 % width).float() * w_step
                point_coords[:, :list_zeros[i], 1] = (point_indices0 // width).float() * h_step

                uncertainty_map1 = uncertainty_map[i:i+1,:,:,:].clone()
                uncertainty_map_zeros = uncertainty_map[i:i+1,:,:,:] * (1-boundry).unsqueeze(1)
                uncertainty_map1[uncertainty_map_zeros[i:i+1,:,:,:] == 0]=torch.min(uncertainty_map1)-1.0
                uncertainty_map1 = uncertainty_map1.view(1, height * width)

                point_indices1 = uncertainty_map1.topk(num_points-list_zeros[i], dim=1)[1]

                point_coords[:, list_zeros[i]:, 0] = (point_indices1 % width).float() * w_step
                point_coords[:, list_zeros[i]:, 1] = (point_indices1 // width).float() * h_step
                point_indices = torch.cat((point_indices0,point_indices1), dim=1)
            point_coordss.append(point_coords)
            point_indicess.append(point_indices)
        point_coords = torch.cat(point_coordss, dim=0)
        point_indices = torch.cat(point_indicess, dim=0)
        return point_indices, point_coords
