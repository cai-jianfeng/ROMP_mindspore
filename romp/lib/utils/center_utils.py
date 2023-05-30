import mindspore # import torch
from mindspore import ops, nn
import constants
from config import args
import numpy as np
from .cam_utils import convert_cam_params_to_centermap_coords

def denormalize_center(center, size=args().centermap_size):
    center = (center+1)/2*size

    center[center<1] = 1
    center[center>size - 1] = size - 1
    if isinstance(center, np.ndarray):
        center = center.astype(np.int32)
    elif isinstance(center, mindspore.Tensor):
        center = center.long()
    return center

def process_gt_center(center_normed):
    valid_mask = center_normed[:,:,0]>-1
    valid_inds = ops.nonzero(valid_mask)  # ops.nonzero 返回 [dim, 2] 形状的矩阵, 每一行表示不为 0 的坐标 (x, y)
    valid_batch_inds, valid_person_ids = valid_inds[:, 0], valid_inds[:, 1]
    center_gt = ((center_normed+1)/2*args().centermap_size).long()
    center_gt_valid = center_gt[valid_mask]
    return (valid_batch_inds, valid_person_ids, center_gt_valid)


def parse_gt_center3d(cam_mask, cams, size=args().centermap_size):
    batch_ids, person_ids = ops.where(cam_mask)
    cam_params = cams[batch_ids, person_ids]
    centermap_coords = convert_cam_params_to_centermap_coords(cam_params)
    czyxs = denormalize_center(centermap_coords, size=size)
    #sample_view_ids = determine_sample_view(batch_ids,czyxs)
    return batch_ids, person_ids, czyxs