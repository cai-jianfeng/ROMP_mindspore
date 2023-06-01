import mindspore  # import torch
# import mindspore # import torch.nn as nn
from mindspore import ops, nn
import numpy as np

import sys, os
import config
from config import args
import constants
from smpl_family.smpl import SMPL
from utils.projection import vertices_kp3d_projection
from utils.rot_6D import rot6D_to_angular


class SMPLWrapper(nn.Cell):
    def __init__(self):
        super(SMPLWrapper, self).__init__()
        # self.smpl_model = smpl_model.create(args().smpl_model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, \
        #    batch_size=args().batch_size,model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',flat_hand_mean=True, use_pca=False).cuda()
        self.smpl_model = SMPL(args().smpl_model_path, model_type='smpl')
        self.part_name = ['cam', 'global_orient', 'body_pose', 'betas']
        self.part_idx = [args().cam_dim, args().rot_dim, (args().smpl_joint_num - 1) * args().rot_dim, 10]

        self.unused_part_name = ['left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
                                 'expression']
        self.unused_part_idx = [15, 15, 3, 3, 3, 10]

        self.kps_num = 25  # + 21*2
        self.params_num = np.array(self.part_idx).sum().item()
        self.global_orient_nocam = mindspore.Tensor.from_numpy(constants.global_orient_nocam).unsqueeze(0)
        self.joint_mapper_op25 = mindspore.Tensor.from_numpy(
            constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()
        self.joint_mapper_op25 = mindspore.Tensor.from_numpy(
            constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()

    def construct(self, outputs, meta_data):
        idx_list, params_dict = [0], {}
        for i, (idx, name) in enumerate(zip(self.part_idx, self.part_name)):
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = outputs['params_pred'][:, idx_list[i]: idx_list[i + 1]]

        if args().Rot_type == '6D':
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = ops.cat([params_dict['body_pose'], ops.zeros((N, 6))], 1)
        params_dict['poses'] = ops.cat([params_dict['global_orient'], params_dict['body_pose']], 1)

        vertices, joints54_17 = self.smpl_model(betas=params_dict['betas'], poses=params_dict['poses'],
                                                root_align=args().smpl_mesh_root_align)
        vertices, joints54_17 = mindspore.Tensor.from_numpy(vertices.numpy()), mindspore.Tensor.from_numpy(joints54_17.numpy())
        outputs.update({'params': params_dict, 'verts': vertices, 'j3d': joints54_17[:, :54],
                        'joints_h36m17': joints54_17[:, 54:]})

        outputs.update(vertices_kp3d_projection(outputs['j3d'], outputs['params']['cam'],
                                                joints_h36m17_preds=outputs['joints_h36m17'],
                                                vertices=outputs['verts'], input2orgimg_offsets=meta_data['offsets'],
                                                presp=args().perspective_proj))
        print('=============output succeed=================')

        return outputs
