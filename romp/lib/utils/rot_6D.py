import mindspore # import torch
from mindspore import ops, nn
# from torch.nn import functional as F
import numpy as np

def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(
        pred_rotmat.reshape(-1, 3, 3)).reshape(batch_size, -1)
    return pose

def rot6d_to_rotmat_batch(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    # TODO: einsum
    b1 = ops.L2Normalize()(a1)
    b2 = ops.L2Normalize()(a2 - ops.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)

    # inp = a2 - ops.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1
    # denom = inp.pow(2).sum(axis=1).sqrt().unsqueeze(-1) + 1e-8
    # b2 = inp / denom

    b3 = ops.cross(b1, b2)
    return ops.stack((b1, b2, b3), axis=-1)

def rot6d_to_rotmat(x):
    x = x.view(-1,3,2)

    # Normalize the first vector
    b1 = ops.L2Normalize(axis=1, epsilon=1e-6)(x[:, :, 0])

    dot_prod = ops.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = ops.L2Normalize(axis=-1, epsilon=1e-6)(x[:, :, 1] - dot_prod * b1)

    # Finish building the basis by taking the cross product
    b3 = ops.cross(b1, b2, dim=1)
    rot_mats = ops.stack([b1, b2, b3], axis=-1)

    return rot_mats


def batch_rodrigues(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = ops.norm(axisang + 1e-8, p=2, axis=1)
    angle = ops.unsqueeze(axisang_norm, -1)
    axisang_normalized = ops.div(axisang, angle)
    angle = angle * 0.5
    v_cos = ops.cos(angle)
    v_sin = ops.sin(angle)
    quat = ops.cat([v_cos, v_sin * axisang_normalized], axis=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, axis=1, keepaxis=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = ops.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         axis=1).view(batch_size, 3, 3)
    return rotMat

'''
def rotation_matrix_to_angle_axis(
        rotation_matrix: mindspore.Tensor) -> mindspore.Tensor:
    r"""Convert 3x3 rotation matrix to Rodrigues vector.
    Args:
        rotation_matrix (mindspore.Tensor): rotation matrix.
    Returns:
        mindspore.Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = ops.rand(2, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if not isinstance(rotation_matrix, mindspore.Tensor):
        raise TypeError("Input type is not a mindspore.Tensor. Got {}".format(
            type(rotation_matrix)))

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a (*, 3, 3) tensor. Got {}".format(
                rotation_matrix.shape))
    quaternion: mindspore.Tensor = rotation_matrix_to_quaternion(rotation_matrix)
    aa=quaternion_to_angle_axis(quaternion)
    #aa[ops.isnan(aa)] = 0.0
    return aa


def rotation_matrix_to_quaternion(
        rotation_matrix: mindspore.Tensor,
        eps: float = 1e-8) -> mindspore.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.
    The quaternion vector has components in (x, y, z, w) format.
    Args:
        rotation_matrix (mindspore.Tensor): the rotation matrix to convert.
        eps (float): small value to avoid zero division. Default: 1e-8.
    Return:
        mindspore.Tensor: the rotation in quaternion.
    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`
    Example:
        >>> input = ops.rand(4, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not isinstance(rotation_matrix, mindspore.Tensor):
        raise TypeError("Input type is not a mindspore.Tensor. Got {}".format(
            type(rotation_matrix)))

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a (*, 3, 3) tensor. Got {}".format(
                rotation_matrix.shape))

    def safe_zero_division(numerator: mindspore.Tensor,
                           denominator: mindspore.Tensor) -> mindspore.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / ops.clamp(denominator, min=eps)

    rotation_matrix_vec: mindspore.Tensor = rotation_matrix.view(
        *rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(
        rotation_matrix_vec, chunks=9, axis=-1)

    trace: mindspore.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = ops.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return ops.cat([qx, qy, qz, qw], axis=-1)

    def cond_1():
        sq = ops.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return ops.cat([qx, qy, qz, qw], axis=-1)

    def cond_2():
        sq = ops.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return ops.cat([qx, qy, qz, qw], axis=-1)

    def cond_3():
        sq = ops.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return ops.cat([qx, qy, qz, qw], axis=-1)

    where_2 = ops.where(m11 > m22, cond_2(), cond_3())
    where_1 = ops.where(
        (m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: mindspore.Tensor = ops.where(
        trace > 0., trace_positive_cond(), where_1)
    return quaternion


def normalize_quaternion(quaternion: mindspore.Tensor,
                         eps: float = 1e-12) -> mindspore.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (x, y, z, w) format.
    Args:
        quaternion (mindspore.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.
    Return:
        mindspore.Tensor: the normalized quaternion of shape :math:`(*, 4)`.
    Example:
        >>> quaternion = mindspore.Tensor([1., 0., 1., 0.])
        >>> normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, mindspore.Tensor):
        raise TypeError("Input type is not a mindspore.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    return F.normalize(quaternion, p=2, axis=-1, eps=eps)


# based on:
# https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247

def quaternion_to_rotation_matrix(quaternion: mindspore.Tensor) -> mindspore.Tensor:
    r"""Converts a quaternion to a rotation matrix.
    The quaternion should be in (x, y, z, w) format.
    Args:
        quaternion (mindspore.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 4)`.
    Return:
        mindspore.Tensor: the rotation matrix of shape :math:`(*, 3, 3)`.
    Example:
        >>> quaternion = mindspore.Tensor([0., 0., 1., 0.])
        >>> quaternion_to_rotation_matrix(quaternion)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, mindspore.Tensor):
        raise TypeError("Input type is not a mindspore.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    # normalize the input quaternion
    quaternion_norm: mindspore.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, axis=-1)

    # compute the actual conversion
    tx: mindspore.Tensor = 2.0 * x
    ty: mindspore.Tensor = 2.0 * y
    tz: mindspore.Tensor = 2.0 * z
    twx: mindspore.Tensor = tx * w
    twy: mindspore.Tensor = ty * w
    twz: mindspore.Tensor = tz * w
    txx: mindspore.Tensor = tx * x
    txy: mindspore.Tensor = ty * x
    txz: mindspore.Tensor = tz * x
    tyy: mindspore.Tensor = ty * y
    tyz: mindspore.Tensor = tz * y
    tzz: mindspore.Tensor = tz * z
    one: mindspore.Tensor = mindspore.Tensor(1.)

    matrix: mindspore.Tensor = ops.stack([
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy)
    ], axis=-1).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, axis=0)
    return matrix


def quaternion_to_angle_axis(quaternion: mindspore.Tensor) -> mindspore.Tensor:
    """Convert quaternion vector to angle axis of rotation.
    The quaternion should be in (x, y, z, w) format.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (mindspore.Tensor): tensor with quaternions.
    Return:
        mindspore.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = ops.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not ops.is_tensor(quaternion):
        raise TypeError("Input type is not a mindspore.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(
                quaternion.shape))
    # unpack input and compute conversion
    q1: mindspore.Tensor = quaternion[..., 1]
    q2: mindspore.Tensor = quaternion[..., 2]
    q3: mindspore.Tensor = quaternion[..., 3]
    sin_squared_theta: mindspore.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: mindspore.Tensor = ops.sqrt(sin_squared_theta)
    cos_theta: mindspore.Tensor = quaternion[..., 0]
    two_theta: mindspore.Tensor = 2.0 * ops.where(
        cos_theta < 0.0, ops.atan2(-sin_theta, -cos_theta),
        ops.atan2(sin_theta, cos_theta))

    k_pos: mindspore.Tensor = two_theta / sin_theta
    k_neg: mindspore.Tensor = 2.0 * ops.ones_like(sin_theta)
    k: mindspore.Tensor = ops.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: mindspore.Tensor = ops.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def angle_axis_to_rotation_matrix(angle_axis: mindspore.Tensor) -> mindspore.Tensor:
    r"""Convert 3d vector of axis-angle rotation to 3x3 rotation matrix
    Args:
        angle_axis (mindspore.Tensor): tensor of 3d vector of axis-angle rotations.
    Returns:
        mindspore.Tensor: tensor of 3x3 rotation matrices.
    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 3, 3)`
    Example:
        >>> input = ops.rand(1, 3)  # Nx3
        >>> output = angle_axis_to_rotation_matrix(input)  # Nx3x3
    """
    if not isinstance(angle_axis, mindspore.Tensor):
        raise TypeError("Input type is not a mindspore.Tensor. Got {}".format(
            type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError(
            "Input size must be a (*, 3) tensor. Got {}".format(
                angle_axis.shape))

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = ops.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, axis=1)
        cos_theta = ops.cos(theta)
        sin_theta = ops.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = ops.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, axis=1)
        k_one = ops.ones_like(rx)
        rotation_matrix = ops.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], axis=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = ops.unsqueeze(angle_axis, axis=1)
    theta2 = ops.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, axis=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).astype(theta2)
    mask_neg = (mask == False).astype(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = ops.eye(3).to(angle_axis.device).astype(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).tile(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4

'''

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = ops.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[ops.isnan(aa)] = 0.0
    return aa

def quaternion_to_angle_axis(quaternion: mindspore.Tensor) -> mindspore.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (mindspore.Tensor): tensor with quaternions.

    Return:
        mindspore.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = ops.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not ops.is_tensor(quaternion):
        raise TypeError("Input type is not a mindspore.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: mindspore.Tensor = quaternion[..., 1]
    q2: mindspore.Tensor = quaternion[..., 2]
    q3: mindspore.Tensor = quaternion[..., 3]
    sin_squared_theta: mindspore.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: mindspore.Tensor = ops.sqrt(sin_squared_theta)
    cos_theta: mindspore.Tensor = quaternion[..., 0]
    two_theta: mindspore.Tensor = 2.0 * ops.where(
        cos_theta < 0.0,
        ops.atan2(-sin_theta, -cos_theta),
        ops.atan2(sin_theta, cos_theta))

    k_pos: mindspore.Tensor = two_theta / sin_theta
    k_neg: mindspore.Tensor = 2.0 * ops.ones_like(sin_theta)
    k: mindspore.Tensor = ops.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: mindspore.Tensor = ops.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = ops.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not ops.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a mindspore.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = ops.swapaxes(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = ops.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.tile((4, 1)).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = ops.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.tile((4, 1)).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = ops.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.tile((4, 1)).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = ops.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.tile((4, 1)).t()

    mask_c0 = mask_d2.astype(mindspore.int32) * mask_d0_d1.astype(mindspore.int32)
    mask_c1 = mask_d2.astype(mindspore.int32) * (~mask_d0_d1).astype(mindspore.int32)
    mask_c2 = (~mask_d2).astype(mindspore.int32) * mask_d0_nd1.astype(mindspore.int32)
    mask_c3 = (~mask_d2).astype(mindspore.int32) * (~mask_d0_nd1).astype(mindspore.int32)
    mask_c0 = mask_c0.view(-1, 1).astype(q0.dtype)
    mask_c1 = mask_c1.view(-1, 1).astype(q1.dtype)
    mask_c2 = mask_c2.view(-1, 1).astype(q2.dtype)
    mask_c3 = mask_c3.view(-1, 1).astype(q3.dtype)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q = q / ops.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3) * 0.5  # noqa
    return q


if __name__ == '__main__':
    pose = rot6D_to_angular(ops.rand(16, 6))
    print(pose.shape)
    print(pose[0])
