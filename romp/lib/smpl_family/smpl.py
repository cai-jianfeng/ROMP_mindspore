from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
sys.path.append('/home/jianfeng_intern/ROMP/ROMP-master/romp/lib')
print(sys.path)
import os.path as osp
import mindspore as ms
from mindspore import nn, ops
import pickle
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F


class VertexJointSelector(nn.Cell):
    def __init__(self, extra_joints_idxs, J_regressor_extra9, J_regressor_h36m17, dtype=ms.float32):
        super(VertexJointSelector, self).__init__()
        self._set_attr_for_tensor('extra_joints_idxs', extra_joints_idxs)
        self._set_attr_for_tensor('J_regressor_extra9', J_regressor_extra9)
        self._set_attr_for_tensor('J_regressor_h36m17', J_regressor_h36m17)

    def construct(self, vertices, joints):
        extra_joints21 = ops.index_select(vertices, 1, self.extra_joints_idxs)
        b, i, k = vertices.shape
        j, _ = self.J_regressor_extra9.shape
        # extra_joints9 = ops.einsum('bik,ji->bjk', [vertices, self.J_regressor_extra9])
        vertices_1 = vertices.transpose((0, 2, 1)).view(-1, vertices.shape[1])
        J_regressor_extra9_1 = self.J_regressor_extra9.transpose((1, 0))
        extra_joints9 = ops.matmul(vertices_1, J_regressor_extra9_1).view(b, k, j).transpose(0, 2, 1)
        b, i, k = vertices.shape
        j, _ = self.J_regressor_h36m17.shape
        # joints_h36m17 = ops.einsum('bik,ji->bjk', [vertices, self.J_regressor_h36m17])
        J_regressor_h36m17_1 = self.J_regressor_h36m17.transpose((1, 0))
        joints_h36m17 = ops.matmul(vertices_1, J_regressor_h36m17_1).view(b, k, j).transpose(0, 2, 1)
        # 54 joints = 24 smpl joints + 21 face & feet & hands joints + 9 extra joints from different datasets + 17 joints from h36m
        joints54_17 = ops.cat([joints, extra_joints21, extra_joints9, joints_h36m17], axis=1)
        # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
        # joints_h36m17_pelvis = joints_h36m17[:,14].unsqueeze(1)
        # joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        return joints54_17


class SMPL(nn.Cell):
    def __init__(self, model_path, model_type='smpl', dtype=ms.float32):
        super(SMPL, self).__init__()
        self.dtype = dtype
        model_info = torch.load(model_path)
        # print(type(model_info))
        for key, item in model_info.items():
            # print(key, '; ', item.shape)
            model_info[key] = ms.Tensor.from_numpy(item.numpy())
        self.vertex_joint_selector = VertexJointSelector(model_info['extra_joints_index'],
                                                         model_info['J_regressor_extra9'],
                                                         model_info['J_regressor_h36m17'], dtype=self.dtype)
        self._set_attr_for_tensor('faces_tensor', model_info['f'])
        # The vertices of the template model
        self._set_attr_for_tensor('v_template', model_info['v_template'])
        # The shape components, take the top 10 PCA componence.
        if model_type == 'smpl':
            self._set_attr_for_tensor('shapedirs', model_info['shapedirs'])
        elif model_type == 'smpla':
            self._set_attr_for_tensor('shapedirs', model_info['smpla_shapedirs'])

        self._set_attr_for_tensor('J_regressor', model_info['J_regressor'])
        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207, then transpose to 207 x 6890*3
        self._set_attr_for_tensor('posedirs', model_info['posedirs'])
        # indices of parents for each joints
        self._set_attr_for_tensor('parents', model_info['kintree_table'])
        self._set_attr_for_tensor('lbs_weights', model_info['weights'])

    # @time_cost('SMPL')
    def construct(self, betas=None, poses=None, root_align=True, **kwargs):
        """ Forward pass for the SMPL model
            Parameters
            ----------
            betas: ms.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: ms.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            Return
            ----------
            outputs: dict, {'verts': vertices of body meshes, (B x 6890 x 3),
                            'joints54': 54 joints of body meshes, (B x 54 x 3), }
                            #'joints_h36m17': 17 joints of body meshes follow h36m skeleton format, (B x 17 x 3)}
        """
        if isinstance(betas, np.ndarray):
            betas = ms.Tensor.from_numpy(betas).astype(self.dtype)

        if isinstance(poses, np.ndarray):
            poses = ms.Tensor.from_numpy(poses).astype(self.dtype)
        vertices, joints = lbs(betas, poses, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)
        # print(vertices.shape, '; ', joints.shape)  # (1, 6890, 3) ;  (1, 24, 3)
        # vertices = vertices.astype(joints.dtype)
        # print(vertices.dtype, '; ', joints.dtype)
        joints54_17 = self.vertex_joint_selector(vertices, joints)
        if root_align:
            # use the Pelvis of most 2D image, not the original Pelvis
            root_trans = joints54_17[:, [45, 46]].mean(1).unsqueeze(1)
            joints54_17 = joints54_17 - root_trans
            vertices = vertices - root_trans

        return vertices, joints54_17


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, dtype=ms.float32):
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : ms.tensor BxNB
            The tensor of shape parameters
        pose : ms.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template ms.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : ms.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : ms.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : ms.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: ms.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: ms.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: ms.dtype, optional

        Returns
        -------
        verts: ms.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: ms.tensor BxJx3
            The joints of the model
    """
    # print(betas.dtype, '; ', pose.dtype)
    batch_size = betas.shape[0]
    # Add shape contribution
    # print(betas.shape, '; ', shapedirs.shape)
    b, l = betas.shape
    m, k, _ = shapedirs.shape
    # b = 1, l = 10, m = 6890, k = 3
    # v_shaped = v_template + ops.einsum('bl,mkl->bmk', [betas, shapedirs])
    shapedirs_1 = shapedirs.view((-1, shapedirs.shape[2])).transpose((1, 0))
    # print(shapedirs_1.shape)  # l, mk
    v_shaped = ops.matmul(betas, shapedirs_1).view(b, m, k) + v_template
    # Get the joints
    # NxJx3 array
    # print(v_shaped.shape, '; ', J_regressor.shape)
    b, i, k = v_shaped.shape
    j, _ = J_regressor.shape
    # J = ops.einsum('bik,ji->bjk', [v_shaped, J_regressor])
    v_shaped_1 = v_shaped.transpose((0, 2, 1)).view(-1, v_shaped.shape[1])
    J_regressor_1 = J_regressor.transpose((1, 0))
    J = ops.matmul(v_shaped_1, J_regressor_1).view(b, k, j).transpose(0, 2, 1)
    # print(J.shape)
    dtype = pose.dtype
    # dtype = torch.float32
    posedirs = posedirs.astype(dtype)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = ops.eye(3, dtype=dtype)
    # print('00001')
    rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view((batch_size, -1, 3, 3)).astype(dtype)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view((batch_size, -1)).astype(dtype)
    # (N x P) x (P, V * 3) -> N x V x 3
    pose_offsets = ops.matmul(pose_feature, posedirs.astype(dtype)).view((batch_size, -1, 3))
    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand(ms.Tensor([batch_size, -1, -1]))
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = ops.matmul(W, A.view((batch_size, num_joints, 16))).view((batch_size, -1, 4, 4))

    homogen_coord = ops.ones([batch_size, v_posed.shape[1], 1], dtype=dtype)
    v_posed_homo = ops.cat([v_posed, homogen_coord], axis=2)
    v_homo = ops.matmul(T, ops.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=ms.float32):
    """ Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: ms.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: ms.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    angle = ops.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = ops.unsqueeze(ops.cos(angle), dim=1)
    sin = ops.unsqueeze(ops.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = ops.split(rot_dir, 1, axis=1)
    K = ops.zeros((batch_size, 3, 3), dtype=dtype)

    zeros = ops.zeros((batch_size, 1), dtype=dtype)
    # K = ops.cat([zeros, -rz], dim=1)  # .view((batch_size, 3, 3))
    # print('000000')
    # K = ops.cat([K, ry, rz], dim=1)
    # print('000001')
    # K = ops.cat([K, zeros, -rx, -ry, rx, zeros], dim=1)
    # print('000002')
    # K = ops.cat([K, rx, zeros], dim=1).view((batch_size, 3, 3))
    K = ops.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1).view((batch_size, 3, 3))
    ident = ops.eye(3, dtype=dtype).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * ops.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return ops.cat([ops.pad(R, [0, 0, 0, 1]),
                    ops.pad(t, [0, 0, 0, 1], value=1)], axis=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=ms.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : ms.tensor BxNx3x3
        Tensor of rotation matrices
    joints : ms.tensor BxNx3
        Locations of joints
    parents : ms.tensor BxN
        The kinematic tree of each object
    dtype : ms.dtype, optional:
        The data type of the created tensors, the default is ms.float32

    Returns
    -------
    posed_joints : ms.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : ms.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = ops.unsqueeze(joints, dim=-1)
    rel_joints = joints.copy()  # .astype(rot_mats.dtype)
    rot_mats = rot_mats.astype(rel_joints.dtype)
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    # print(rot_mats.dtype, '; ', rel_joints.dtype)
    transforms_mat = transform_mat(
        rot_mats.reshape((-1, 3, 3)),
        rel_joints.reshape((-1, 3, 1))).reshape((-1, joints.shape[1], 4, 4))
    transform_chain = [transforms_mat[:, 0]]
    # print(transform_chain)
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        # print(transform_chain[parents[i]].dtype, '; ', transforms_mat[:, i].dtype)
        curr_res = ops.matmul(transform_chain[parents[i]],
                              transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = ops.stack(transform_chain, axis=1)
    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = ops.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - ops.pad(ops.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def export_smpl_to_onnx_dynamic(smpl_model, save_file, bs=1):
    "support dynamics batch size but slow"
    # a = ops.rand([bs, 10])
    # b = ops.rand([bs, 72])
    # dynamic_axes = {'smpl_betas': [0], 'smpl_thetas': [0], 'verts': [0], 'joints': [0]}
    # torch.onnx.export(smpl_model, (a, b),
    #                   save_file,
    #                   input_names=['smpl_betas', 'smpl_thetas'],
    #                   output_names=['verts', 'joints', 'faces'],
    #                   export_params=True,
    #                   opset_version=12,
    #                   do_constant_folding=True,
    #                   dynamic_axes=dynamic_axes)
    # print('SMPL onnx saved into: ', save_file)
    pass


def export_smpl_to_onnx_static(smpl_model, save_file, bs=1):
    # a = torch.rand([bs, 10]).cuda()
    # b = torch.rand([bs, 72]).cuda()
    # torch.onnx.export(smpl_model, (a, b),
    #                   save_file,
    #                   input_names=['smpl_betas', 'smpl_thetas'],
    #                   output_names=['verts', 'joints', 'faces'],
    #                   export_params=True,
    #                   opset_version=12,
    #                   do_constant_folding=True)
    # print('SMPL onnx saved into: ', save_file)
    pass


def test_smpl(smpl_model, dtype=ms.float32):
    import time, cv2
    # from visualization import render_human_mesh
    cost_time = []
    batch_size = 1
    a = ops.zeros((batch_size, 10)).astype(dtype)  # .cuda()
    b = ops.zeros((batch_size, 72)).astype(dtype)  # .cuda()
    image_length = 1024
    bg_image = np.ones((image_length, image_length, 3), dtype=np.uint8) * 255
    for _ in range(200):
        start_time = time.time()
        outputs = smpl_model(a, b)
        print(outputs, outputs[0].shape)
        verts_np = (outputs[0].numpy() * image_length / 2).astype(np.float32) + + np.array(
            [[[.5, .5, 0]]]).astype(np.float32) * image_length
        faces_np = outputs[2].cpu().numpy().astype(np.int32)
        rendered_image = render_human_mesh(bg_image, verts_np, faces_np)
        print(rendered_image.shape)
        cv2.imshow('rendering', rendered_image)
        cv2.waitKey(1)
        end_time = time.time()
        cost_time.append(end_time - start_time)
    print('cost time ', np.mean(cost_time))
    print(cost_time[:10])
    # for key, item in outputs.items():
    #    print(key, item.shape)
    return


def test_onnx(dtype=np.float32, batch_size=1):
    smpl_onnx_path = "smpl.onnx"
    # import onnx, onnxruntime
    # onnx_model = onnx.load(smpl_onnx_path)
    # onnx.checker.check_model(onnx_model)
    # ort_session = onnxruntime.InferenceSession(smpl_onnx_path)

    import time
    cost_time = []

    a = np.random.random([batch_size, 10]).astype(dtype)
    b = np.random.random([batch_size, 72]).astype(dtype)

    ort_inputs = {'smpl_betas': a, 'smpl_thetas': b}
    for _ in range(200):
        start_time = time.time()
        # ort_outs = ort_session.run(None, ort_inputs)
        end_time = time.time()
        cost_time.append(end_time - start_time)
    print('cost time ', np.mean(cost_time))
    print(cost_time[:10])


def prepare_smpl_model(dtype):
    model_path = '/home/jianfeng_intern/ROMP/ROMP-master/model_data/smpl_models/smpl_packed_info.pth'
    smpl_model = SMPL(model_path, dtype=dtype)  # .cuda()
    smpl_model.set_train(False)
    return smpl_model


if __name__ == '__main__':
    # test_onnx(batch_size=1)
    dtype = ms.float32
    smpl_model = prepare_smpl_model(dtype)
    test_smpl(smpl_model)
    # export_smpl_to_onnx_static(smpl_model, 'smpl.onnx', bs=1)