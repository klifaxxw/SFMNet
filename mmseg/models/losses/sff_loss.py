import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import Tensor
from typing import Optional, List

from ..builder import LOSSES
from .functional import label_smoothed_nll_loss, soft_dice_score


def to_tensor(x, dtype=None) -> torch.Tensor:
    """Convert input to torch.Tensor."""
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind not in {"O", "M", "U", "S"}:
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    raise ValueError(f"Unsupported input type {type(x)}")


# ===================== Weighted + Joint Loss =====================

class WeightedLoss(_Loss):
    """Wrap loss with a fixed weight."""
    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """Combine two losses as weighted sum."""
    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


# ===================== Dice Loss =====================

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"

class DiceLoss(_Loss):
    """Dice Loss for segmentation."""
    def __init__(
        self,
        mode: str = MULTICLASS_MODE,
        classes: Optional[List[int]] = None,
        log_loss=False,
        from_logits=True,
        smooth: float = 0.0,
        ignore_index=None,
        eps=1e-7,
    ):
        assert mode in {BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE}
        super().__init__()
        self.mode = mode
        self.from_logits = from_logits
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps
        self.log_loss = log_loss
        self.classes = to_tensor(classes, dtype=torch.long) if classes is not None else None

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        bs = y_true.size(0)
        num_classes = y_pred.size(1)

        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        elif self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes).permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes).permute(0, 2, 1)

        elif self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        loss = -torch.log(scores.clamp_min(self.eps)) if self.log_loss else 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()

class SoftCrossEntropyLoss(nn.Module):
    """Label smoothed cross-entropy loss."""
    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )

@LOSSES.register_module()
class SFFLoss(nn.Module):
    """Soft CrossEntropy + Dice Loss for MMseg integration."""
    def __init__(self, ignore_index=255, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = "loss_sff"

        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            first_weight=1.0,
            second_weight=1.0
        )

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, preds, target, **kwargs):
        loss = self.main_loss(preds, target)
        return self.loss_weight * loss
