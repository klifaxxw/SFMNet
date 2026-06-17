# # Copyright (c) OpenMMLab. All rights reserved.
# from collections import OrderedDict

# import mmcv
# import numpy as np
# import torch
# from tqdm import tqdm
# import cv2



# def mask_to_boundary(mask: np.ndarray, dilation_ratio=0.02):
#     """把类别 mask 转成“边界 mask”（类别 id 保留在边界上）。

#     Args:
#         mask (ndarray): HxW，整型类别图。
#         dilation_ratio (float): 边界厚度占对角线比例，和 detectron2 一样默认 0.02。
#     """
#     assert mask.ndim == 2, 'mask_to_boundary expects a 2D mask'
#     h, w = mask.shape
#     diag_len = np.sqrt(h**2 + w**2)
#     dilation = max(1, int(round(dilation_ratio * diag_len)))
#     kernel = np.ones((3, 3), dtype=np.uint8)

#     # 防止腐蚀时把边缘搞没，四周 pad 一圈
#     padded_mask = cv2.copyMakeBorder(
#         mask.astype(np.uint8), 1, 1, 1, 1,
#         cv2.BORDER_CONSTANT, value=0)

#     eroded_with_pad = cv2.erode(padded_mask, kernel, iterations=dilation)
#     eroded = eroded_with_pad[1:-1, 1:-1]

#     # 与 detectron2 一致：boundary = mask - eroded
#     boundary = mask.astype(np.int32) - eroded.astype(np.int32)
#     boundary[boundary != 0] = mask[boundary != 0]  # 保留类别 id
#     boundary = boundary.astype(mask.dtype)
#     return boundary


# def f_score(precision, recall, beta=1):
#     """calculate the f-score value.

#     Args:
#         precision (float | torch.Tensor): The precision value.
#         recall (float | torch.Tensor): The recall value.
#         beta (int): Determines the weight of recall in the combined score.
#             Default: False.

#     Returns:
#         [torch.tensor]: The f-score value.
#     """
#     score = (1 + beta**2) * (precision * recall) / (
#         (beta**2 * precision) + recall)
#     return score


# def intersect_and_union(pred_label,
#                         label,
#                         num_classes,
#                         ignore_index,
#                         label_map=dict(),
#                         reduce_zero_label=False):
#     """Calculate intersection and Union.

#     Args:
#         pred_label (ndarray | str): Prediction segmentation map
#             or predict result filename.
#         label (ndarray | str): Ground truth segmentation map
#             or label filename.
#         num_classes (int): Number of categories.
#         ignore_index (int): Index that will be ignored in evaluation.
#         label_map (dict): Mapping old labels to new labels. The parameter will
#             work only when label is str. Default: dict().
#         reduce_zero_label (bool): Whether ignore zero label. The parameter will
#             work only when label is str. Default: False.

#      Returns:
#          torch.Tensor: The intersection of prediction and ground truth
#             histogram on all classes.
#          torch.Tensor: The union of prediction and ground truth histogram on
#             all classes.
#          torch.Tensor: The prediction histogram on all classes.
#          torch.Tensor: The ground truth histogram on all classes.
#     """

#     if isinstance(pred_label, str):
#         pred_label = torch.from_numpy(np.load(pred_label))
#     else:
#         pred_label = torch.from_numpy((pred_label))

#     if isinstance(label, str):
#         label = torch.from_numpy(
#             mmcv.imread(label, flag='unchanged', backend='pillow'))
#     else:
#         label = torch.from_numpy(label)

#     if label_map is not None:
#         for old_id, new_id in label_map.items():
#             label[label == old_id] = new_id
#     if reduce_zero_label:
#         label[label == 0] = 255
#         label = label - 1
#         label[label == 254] = 255

#     mask = (label != ignore_index)
#     pred_label = pred_label[mask]
#     label = label[mask]
#     intersect = pred_label[pred_label == label]
#     area_intersect = torch.histc(
#         intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
#     area_pred_label = torch.histc(
#         pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
#     area_label = torch.histc(
#         label.float(), bins=(num_classes), min=0, max=num_classes - 1)
#     area_union = area_pred_label + area_label - area_intersect
#     return area_intersect, area_union, area_pred_label, area_label


# def total_intersect_and_union(results,
#                               gt_seg_maps,
#                               num_classes,
#                               ignore_index,
#                               label_map=dict(),
#                               reduce_zero_label=False):
#     """Calculate Total Intersection and Union.

#     Args:
#         results (list[ndarray] | list[str]): List of prediction segmentation
#             maps or list of prediction result filenames.
#         gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
#             truth segmentation maps or list of label filenames.
#         num_classes (int): Number of categories.
#         ignore_index (int): Index that will be ignored in evaluation.
#         label_map (dict): Mapping old labels to new labels. Default: dict().
#         reduce_zero_label (bool): Whether ignore zero label. Default: False.

#      Returns:
#          ndarray: The intersection of prediction and ground truth histogram
#              on all classes.
#          ndarray: The union of prediction and ground truth histogram on all
#              classes.
#          ndarray: The prediction histogram on all classes.
#          ndarray: The ground truth histogram on all classes.
#     """
#     total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
#     total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
#     total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
#     total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
#     for result, gt_seg_map in tqdm(zip(results, gt_seg_maps), total=len(results)):
#         area_intersect, area_union, area_pred_label, area_label = \
#             intersect_and_union(
#                 result, gt_seg_map, num_classes, ignore_index,
#                 label_map, reduce_zero_label)
#         total_area_intersect += area_intersect
#         total_area_union += area_union
#         total_area_pred_label += area_pred_label
#         total_area_label += area_label
#     return total_area_intersect, total_area_union, total_area_pred_label, \
#         total_area_label


# def mean_iou(results,
#              gt_seg_maps,
#              num_classes,
#              ignore_index,
#              nan_to_num=None,
#              label_map=dict(),
#              reduce_zero_label=False):
#     """Calculate Mean Intersection and Union (mIoU)

#     Args:
#         results (list[ndarray] | list[str]): List of prediction segmentation
#             maps or list of prediction result filenames.
#         gt_seg_maps (list[ndarray] | list[str]): list of ground truth
#             segmentation maps or list of label filenames.
#         num_classes (int): Number of categories.
#         ignore_index (int): Index that will be ignored in evaluation.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.
#         label_map (dict): Mapping old labels to new labels. Default: dict().
#         reduce_zero_label (bool): Whether ignore zero label. Default: False.

#      Returns:
#         dict[str, float | ndarray]:
#             <aAcc> float: Overall accuracy on all images.
#             <Acc> ndarray: Per category accuracy, shape (num_classes, ).
#             <IoU> ndarray: Per category IoU, shape (num_classes, ).
#     """
#     iou_result = eval_metrics(
#         results=results,
#         gt_seg_maps=gt_seg_maps,
#         num_classes=num_classes,
#         ignore_index=ignore_index,
#         metrics=['mIoU'],
#         nan_to_num=nan_to_num,
#         label_map=label_map,
#         reduce_zero_label=reduce_zero_label)
#     return iou_result


# def mean_dice(results,
#               gt_seg_maps,
#               num_classes,
#               ignore_index,
#               nan_to_num=None,
#               label_map=dict(),
#               reduce_zero_label=False):
#     """Calculate Mean Dice (mDice)

#     Args:
#         results (list[ndarray] | list[str]): List of prediction segmentation
#             maps or list of prediction result filenames.
#         gt_seg_maps (list[ndarray] | list[str]): list of ground truth
#             segmentation maps or list of label filenames.
#         num_classes (int): Number of categories.
#         ignore_index (int): Index that will be ignored in evaluation.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.
#         label_map (dict): Mapping old labels to new labels. Default: dict().
#         reduce_zero_label (bool): Whether ignore zero label. Default: False.

#      Returns:
#         dict[str, float | ndarray]: Default metrics.
#             <aAcc> float: Overall accuracy on all images.
#             <Acc> ndarray: Per category accuracy, shape (num_classes, ).
#             <Dice> ndarray: Per category dice, shape (num_classes, ).
#     """

#     dice_result = eval_metrics(
#         results=results,
#         gt_seg_maps=gt_seg_maps,
#         num_classes=num_classes,
#         ignore_index=ignore_index,
#         metrics=['mDice'],
#         nan_to_num=nan_to_num,
#         label_map=label_map,
#         reduce_zero_label=reduce_zero_label)
#     return dice_result


# def mean_fscore(results,
#                 gt_seg_maps,
#                 num_classes,
#                 ignore_index,
#                 nan_to_num=None,
#                 label_map=dict(),
#                 reduce_zero_label=False,
#                 beta=1):
#     """Calculate Mean Intersection and Union (mIoU)

#     Args:
#         results (list[ndarray] | list[str]): List of prediction segmentation
#             maps or list of prediction result filenames.
#         gt_seg_maps (list[ndarray] | list[str]): list of ground truth
#             segmentation maps or list of label filenames.
#         num_classes (int): Number of categories.
#         ignore_index (int): Index that will be ignored in evaluation.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.
#         label_map (dict): Mapping old labels to new labels. Default: dict().
#         reduce_zero_label (bool): Whether ignore zero label. Default: False.
#         beta (int): Determines the weight of recall in the combined score.
#             Default: False.


#      Returns:
#         dict[str, float | ndarray]: Default metrics.
#             <aAcc> float: Overall accuracy on all images.
#             <Fscore> ndarray: Per category recall, shape (num_classes, ).
#             <Precision> ndarray: Per category precision, shape (num_classes, ).
#             <Recall> ndarray: Per category f-score, shape (num_classes, ).
#     """
#     fscore_result = eval_metrics(
#         results=results,
#         gt_seg_maps=gt_seg_maps,
#         num_classes=num_classes,
#         ignore_index=ignore_index,
#         metrics=['mFscore'],
#         nan_to_num=nan_to_num,
#         label_map=label_map,
#         reduce_zero_label=reduce_zero_label,
#         beta=beta)
#     return fscore_result


# def eval_metrics(results,
#                  gt_seg_maps,
#                  num_classes,
#                  ignore_index,
#                  metrics=['mIoU'],
#                  nan_to_num=None,
#                  label_map=dict(),
#                  reduce_zero_label=False,
#                  beta=1):
#     """Calculate evaluation metrics
#     Args:
#         results (list[ndarray] | list[str]): List of prediction segmentation
#             maps or list of prediction result filenames.
#         gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
#             truth segmentation maps or list of label filenames.
#         num_classes (int): Number of categories.
#         ignore_index (int): Index that will be ignored in evaluation.
#         metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.
#         label_map (dict): Mapping old labels to new labels. Default: dict().
#         reduce_zero_label (bool): Whether ignore zero label. Default: False.
#      Returns:
#         float: Overall accuracy on all images.
#         ndarray: Per category accuracy, shape (num_classes, ).
#         ndarray: Per category evaluation metrics, shape (num_classes, ).
#     """

#     total_area_intersect, total_area_union, total_area_pred_label, \
#         total_area_label = total_intersect_and_union(
#             results, gt_seg_maps, num_classes, ignore_index, label_map,
#             reduce_zero_label)
#     ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
#                                         total_area_pred_label,
#                                         total_area_label, metrics, nan_to_num,
#                                         beta)

#     return ret_metrics


# def pre_eval_to_metrics(pre_eval_results,
#                         metrics=['mIoU'],
#                         nan_to_num=None,
#                         beta=1):
#     """Convert pre-eval results to metrics.

#     Args:
#         pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
#             for computing evaluation metric
#         metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.
#      Returns:
#         float: Overall accuracy on all images.
#         ndarray: Per category accuracy, shape (num_classes, ).
#         ndarray: Per category evaluation metrics, shape (num_classes, ).
#     """

#     # convert list of tuples to tuple of lists, e.g.
#     # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
#     # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
#     pre_eval_results = tuple(zip(*pre_eval_results))
#     assert len(pre_eval_results) == 4

#     total_area_intersect = sum(pre_eval_results[0])
#     total_area_union = sum(pre_eval_results[1])
#     total_area_pred_label = sum(pre_eval_results[2])
#     total_area_label = sum(pre_eval_results[3])

#     ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
#                                         total_area_pred_label,
#                                         total_area_label, metrics, nan_to_num,
#                                         beta)

#     return ret_metrics


# def total_area_to_metrics(total_area_intersect,
#                           total_area_union,
#                           total_area_pred_label,
#                           total_area_label,
#                           metrics=['mIoU'],
#                           nan_to_num=None,
#                           beta=1):
#     """Calculate evaluation metrics
#     Args:
#         total_area_intersect (ndarray): The intersection of prediction and
#             ground truth histogram on all classes.
#         total_area_union (ndarray): The union of prediction and ground truth
#             histogram on all classes.
#         total_area_pred_label (ndarray): The prediction histogram on all
#             classes.
#         total_area_label (ndarray): The ground truth histogram on all classes.
#         metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.
#      Returns:
#         float: Overall accuracy on all images.
#         ndarray: Per category accuracy, shape (num_classes, ).
#         ndarray: Per category evaluation metrics, shape (num_classes, ).
#     """
#     if isinstance(metrics, str):
#         metrics = [metrics]
#     allowed_metrics = ['mIoU', 'mDice', 'mFscore']
#     if not set(metrics).issubset(set(allowed_metrics)):
#         raise KeyError('metrics {} is not supported'.format(metrics))

#     all_acc = total_area_intersect.sum() / total_area_label.sum()
#     ret_metrics = OrderedDict({'aAcc': all_acc})
#     for metric in metrics:
#         if metric == 'mIoU':
#             iou = total_area_intersect / total_area_union
#             acc = total_area_intersect / total_area_label
#             ret_metrics['IoU'] = iou
#             ret_metrics['Acc'] = acc
#         elif metric == 'mDice':
#             dice = 2 * total_area_intersect / (
#                 total_area_pred_label + total_area_label)
#             acc = total_area_intersect / total_area_label
#             ret_metrics['Dice'] = dice
#             ret_metrics['Acc'] = acc
#         elif metric == 'mFscore':
#             precision = total_area_intersect / total_area_pred_label
#             recall = total_area_intersect / total_area_label
#             f_value = torch.tensor(
#                 [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
#             ret_metrics['Fscore'] = f_value
#             ret_metrics['Precision'] = precision
#             ret_metrics['Recall'] = recall

#     ret_metrics = {
#         metric: value.numpy()
#         for metric, value in ret_metrics.items()
#     }
#     if nan_to_num is not None:
#         ret_metrics = OrderedDict({
#             metric: np.nan_to_num(metric_value, nan=nan_to_num)
#             for metric, metric_value in ret_metrics.items()
#         })
#     return ret_metrics
# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from tqdm import tqdm
import cv2
import itertools
import types


def mask_to_boundary(mask: np.ndarray, dilation_ratio=0.02):
    """把类别 mask 转成“边界 mask”（类别 id 保留在边界上）。

    Args:
        mask (ndarray): HxW，整型类别图。
        dilation_ratio (float): 边界厚度占对角线比例，和 detectron2 一样默认 0.02。
    """
    assert mask.ndim == 2, 'mask_to_boundary expects a 2D mask'
    h, w = mask.shape
    diag_len = np.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * diag_len)))
    kernel = np.ones((3, 3), dtype=np.uint8)

    # 防止腐蚀时把边缘搞没，四周 pad 一圈
    padded_mask = cv2.copyMakeBorder(
        mask.astype(np.uint8), 1, 1, 1, 1,
        cv2.BORDER_CONSTANT, value=0)

    eroded_with_pad = cv2.erode(padded_mask, kernel, iterations=dilation)
    eroded = eroded_with_pad[1:-1, 1:-1]

    # 与 detectron2 一致：boundary = mask - eroded
    boundary = mask.astype(np.int32) - eroded.astype(np.int32)
    # 非 0 位置保留原类别 id
    boundary[boundary != 0] = mask[boundary != 0]
    boundary = boundary.astype(mask.dtype)
    return boundary


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in tqdm(zip(results, gt_seg_maps), total=len(results)):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def total_boundary_intersect_and_union(results,
                                       gt_seg_maps,
                                       num_classes,
                                       ignore_index,
                                       label_map=dict(),
                                       reduce_zero_label=False,
                                       dilation_ratio=0.02):
    """Calculate Total Intersection and Union **only on boundaries**.

    Boundary mask 使用 mask_to_boundary 生成，然后复用 intersect_and_union。
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in tqdm(zip(results, gt_seg_maps), total=len(results)):
        # 转成 numpy 再提 boundary
        if isinstance(result, str):
            pred_np = np.load(result)
        else:
            pred_np = np.asarray(result)

        if isinstance(gt_seg_map, str):
            gt_np = mmcv.imread(gt_seg_map, flag='unchanged', backend='pillow')
        else:
            gt_np = np.asarray(gt_seg_map)

        b_pred = mask_to_boundary(pred_np, dilation_ratio=dilation_ratio)
        b_gt = mask_to_boundary(gt_np, dilation_ratio=dilation_ratio)

        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                b_pred, b_gt, num_classes, ignore_index,
                label_map, reduce_zero_label)

        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)"""

    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)"""

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean F-score (mFscore)"""

    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def mean_biou(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Boundary IoU (mBIoU)."""
    biou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mBIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return biou_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):

    

    if isinstance(metrics, str):
        metrics = [metrics]

    # 是否需要算 boundary IoU
    compute_boundary = 'mBIoU' in metrics
    base_metrics = [m for m in metrics if m != 'mBIoU']

    # ------------------------------
    # 情况 1：只计算 mBIoU
    # ------------------------------
    if compute_boundary and len(base_metrics) == 0:
        if isinstance(gt_seg_maps, types.GeneratorType):
            _, gt2 = itertools.tee(gt_seg_maps, 2)
        else:
            gt2 = gt_seg_maps

        b_intersect, b_union, b_pred_label, b_label = total_boundary_intersect_and_union(
            results, gt2, num_classes, ignore_index, label_map, reduce_zero_label)

        b_metrics = total_area_to_metrics(
            b_intersect, b_union, b_pred_label, b_label,
            metrics=['mIoU'], 
            nan_to_num=nan_to_num)

        b_iou = b_metrics['IoU']
        return OrderedDict({
            'BIoU': b_iou,
            
        })

    gt2 = None
    if compute_boundary:
        if isinstance(gt_seg_maps, types.GeneratorType):
            gt_seg_maps, gt2 = itertools.tee(gt_seg_maps, 2)
        else:
            gt2 = gt_seg_maps

    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = \
        total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map, reduce_zero_label)

    ret_metrics = total_area_to_metrics(
        total_area_intersect, total_area_union,
        total_area_pred_label, total_area_label,
        metrics=base_metrics, nan_to_num=nan_to_num, beta=beta)

    if compute_boundary:
        b_intersect, b_union, b_pred_label, b_label = total_boundary_intersect_and_union(
            results, gt2, num_classes, ignore_index, label_map, reduce_zero_label)

        b_metrics = total_area_to_metrics(
            b_intersect, b_union, b_pred_label, b_label,
            metrics=['mIoU'], nan_to_num=nan_to_num)

        b_iou = b_metrics['IoU']
        ret_metrics['BIoU'] = b_iou
    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    注意：pre_eval_results 是基于直方图的累积结果，
    当前实现 **不支持** 从 pre_eval 直接恢复 Boundary IoU，
    因此这里不支持 'mBIoU'。
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    assert 'mBIoU' not in metrics, \
        'mBIoU is not supported in pre_eval_to_metrics. ' \
        'Please use eval_metrics with full results/gt instead.'

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU', 'mDice',
            'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
