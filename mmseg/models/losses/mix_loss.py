# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss
from .cross_entropy_loss import cross_entropy
from .dice_loss import dice_loss, binary_dice_loss
from mmseg.utils import get_root_logger

def sobel_edges(x):
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    edge = (edge > 0).float()
    return edge


# 因为将背景类也作为学习目标，因此这里从1开始
# 否则，应该从0开始
@weighted_loss
def boundary_bce_loss(pred, target, class_weight=None,
                  ignore_index=255):
    B, C, H, W = pred.shape
    total_loss = 0.0
    count = 0
    target = target.permute(0, 3, 1, 2).float()  # B, C, H, W
    # Iterate over each class
    for c in range(1, C):
        if c == ignore_index:
            continue
        edge = sobel_edges(target[:, c:c + 1, :, :])
        pred_c = pred[:, c:c + 1, :, :]
        # If there are edges, calculate BCE loss
        if edge.sum() > 0:
            loss = F.binary_cross_entropy(pred_c * edge, edge, reduction='none')
            if class_weight is not None:
                loss *= class_weight[c]
            total_loss += loss
            count += 1
    count = 1 if count < 2 else count
    return total_loss / count


@LOSSES.register_module()
class MixLoss(nn.Module):
    """
    ce + dice + bd dice
    1 piexl accuracy
    2 area accuracy
    3 boundary accuracy 
    """
    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=[1.0, 0.5],
                 ignore_index=255,
                 loss_name='loss_mix',
                 avg_non_ignore=False,
                 total_iters=None,
                 **kwards):
        super(MixLoss, self).__init__()
        assert isinstance(loss_weight, list), "loss_weight 必须是 list 类型"
        assert len(loss_weight) == 2, "loss_weight 长度必须为 2"

        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self.piexl_criterion = cross_entropy
        self.area_criterion = dice_loss
        # 后续替换结构性损失，暂时不重要
        self.boundary_criterion = boundary_bce_loss
        self.total_iters = total_iters
        self.iter_count = 0
        self.logger = get_root_logger()

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        self.iter_count = self.iter_count + 1
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None
        # Note: 
        ## piexl loss
        piexl_loss = self.loss_weight[0] * self.piexl_criterion(
            pred,
            target,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=self.ignore_index,
            **kwargs)

        ## boundary loss
        num_classes = pred.shape[1]
        valid_mask = (target != self.ignore_index).long()
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        pred_d = torch.sigmoid(pred) 
        boundary_loss = self.boundary_criterion(
            pred_d,
            one_hot_target,
            reduction=reduction,
            avg_factor=avg_factor,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        pred = F.softmax(pred, dim=1) 
        area_loss = self.area_criterion(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        alpha = 1 - (self.iter_count / self.total_iters)
        beta = 1 - alpha
        loss = piexl_loss + self.loss_weight[1] * (alpha * area_loss + beta * boundary_loss)
        if self.iter_count % 100 == 0:
            self.logger.info(
                f"[Iter {self.iter_count}] piexl_loss: {piexl_loss.detach().cpu().item():.4f}, "
                f"area_loss: {area_loss.detach().cpu().item():.4f}, "
                f"boundary_loss: {boundary_loss.detach().cpu().item():.4f}, "
                f"total_loss: {loss.detach().cpu().item():.4f}"
            )
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
