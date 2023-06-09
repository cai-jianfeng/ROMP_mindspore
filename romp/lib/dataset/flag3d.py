import sys, os
sys.path.append(os.path.join(os.getcwd(), 'romp/lib'))

import numpy as np
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs
import joblib
import pickle
import mindspore as ms
default_mode = args().image_loading_mode


def preprocess(kpts):
    kpts[:, 0] = -1 * kpts[:, 0]
    kpts[:, 1] = -1 * kpts[:, 1]
    return kpts


def inverse_symbol(name):
    m = int(name[3])
    p = int(name[6:8])
    a = int(name[10:12])
    r = int(name[15])
    n = (m - 1) * 1800 + (p - 1) * 180 + (a - 1) * 3 + r
    if n % 1800 == 0:
        s = n // 1800
    else:
        s = n // 1800 + 1
    if (n - (s - 1) * 1800) % 300 == 0:
        c = (n - (s - 1) * 1800) // 300
    else:
        c = (n - (s - 1) * 1800) // 300 + 1
    return f"S00{s}C00{c}"


def train_test_split(names):
    train_list = []
    test_list = []
    for i in range(len(names)):
        name = names[i]
        m = int(name[3])
        p = int(name[6:8])
        a = int(name[10:12])
        r = int(name[15])
        n = (m - 1) * 1800 + (p - 1) * 180 + (a - 1) * 3 + r
        if n % 300 <= 59:
            test_list.append(name)
        else:
            train_list.append(name)
    return train_list, test_list


def FLAG3D(base_class=default_mode):
    class FLAG3D(Base_Classes[base_class]):
        def __init__(self, train_flag=False, split='train', mode='vibe', regress_smpl=True, **kwargs):
            super(FLAG3D, self).__init__(train_flag, regress_smpl=regress_smpl)
            self.img_folder = "/home/jianfeng_intern/dataset/images"
            self.kpts_folder_3d = "/home/jianfeng_intern/dataset/smpl_24_joints/"
            self.kpts_folder_2d = "/home/jianfeng_intern/dataset/2d_joints/"
            self.param_folder = "/home/jianfeng_intern/dataset/smpl_param/"
            self.mode = mode
            self.root_inds = [constants.SMPL_ALL_54['Pelvis_SMPL']]
            self.split = split
            self.regress_smpl = regress_smpl
            self.ratio = 10
            self.joint_mapper = constants.joint_mapping(constants.COCO_17, constants.SMPL_ALL_54)
            self.joint3d_mapper = constants.joint_mapping(constants.SMPL_24, constants.SMPL_ALL_54)
            self.smplr = SMPLR(use_gender=True)
            self.load_dataset()

        def load_dataset(self):
            for root, names, _ in os.walk(self.img_folder):
                break
            # print(root, names)
            train_names, test_names = train_test_split(names)
            self.train_names = sorted(train_names)
            self.frames = []
            for i in range(len(train_names)):
                l = len(os.listdir(os.path.join(root, self.train_names[i]))) // self.ratio
                self.frames.append(l)
            self.len = np.sum(self.frames)
            self.cum_frames = np.cumsum(self.frames)

        def __len__(self):
            return self.len

        def get_image_info(self, index):
            subject_ids, genders, kp2ds, kp3ds, params, bbox, valid_mask_2d, valid_mask_3d = [[] for i in range(8)]
            if index <= self.cum_frames[0]:
                seq_id = 0
                frame_id = max(0, index * self.ratio - 1)
            else:
                seq_id = np.where(index > self.cum_frames)[0][-1] + 1
                frame_id = max(0, (index - self.cum_frames[seq_id - 1]) * self.ratio - 1)
            imgpath = os.path.join(os.path.join(self.img_folder, self.train_names[seq_id]),
                                   "{:06}.jpg".format(frame_id))
            image = cv2.imread(imgpath)[:, :, ::-1]

            kp3d = np.load(os.path.join(self.kpts_folder_3d,
                                        inverse_symbol(self.train_names[seq_id])[:4] + "C001" + self.train_names[
                                            seq_id]) + ".npy")[frame_id]
            # print('01')
            kp3d = preprocess(kp3d)
            kp3d = self.map_kps(kp3d, self.joint3d_mapper)
            # print('02')
            f = open(os.path.join(self.kpts_folder_2d, self.train_names[seq_id]) + ".pkl", 'rb')
            kp2d = pickle.load(f, encoding='latin1')["keypoint"][0, frame_id, :, :]
            kp2d_gt = self.map_kps(kp2d, self.joint_mapper)
            # print('03')
            param = joblib.load(os.path.join(self.param_folder,
                                             inverse_symbol(self.train_names[seq_id])[:4] + "C001" + self.train_names[
                                                 seq_id]) + ".pkl")
            # print('04')
            pose_param = param["poses"][frame_id]
            beta_param = param["shapes"][0]
            pose_param[:3] = param["Rh"][frame_id]
            root_trans = np.expand_dims(param["Th"][frame_id], axis=0)
            params.append(np.concatenate([pose_param[:66], beta_param[:10]]))
            if self.train_names[seq_id][3] == "3":
                gender = "f"
            else:
                gender = "m"

            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            valid_mask_2d.append([True, True, False])
            valid_mask_3d.append([True, False, False, False, False, False])
            verts = None

            subject_ids.append(0)
            genders.append(gender)
            # root_trans=None
            kp3ds.append(kp3d)
            kp2ds.append(kp2d_gt)
            valid_mask_2d, valid_mask_3d, params = np.array(valid_mask_2d), np.array(valid_mask_3d), np.array(params)
            if self.regress_smpl:
                verts = []
                poses, betas = np.concatenate([params[:, :-10], np.zeros((len(params), 6))], 1), params[:, -10:]
                for pose, beta, gender in zip(poses, betas, genders):
                    gender = 'n' if gender is None else gender
                    # TODO: 这里将 pose 与 beta 修改成 ms.Tensor 的形式是因为 ms.Cell 类型的网络不支持除 ms.Tensor/Parameter 和基本类型(int等) 以外的输入
                    pose, beta = ms.Tensor.from_numpy(pose), ms.Tensor.from_numpy(beta)
                    verts.append(self.smplr(pose, beta, gender)[0])
                    # smpl_outs = self.smplr(pose, beta, gender)
                    # kp3ds.append(smpl_outs['j3d'].numpy())
                    # verts.append(smpl_outs['verts'].numpy())
                # kp3ds = np.concatenate(kp3ds, 0)
                verts = np.concatenate(verts, 0)

            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': subject_ids,
                        'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d,
                        'kp3ds': kp3ds, 'params': params, 'root_trans': root_trans, 'verts': verts,
                        'img_size': image.shape[:2], 'ds': "flag3d"}
            return img_info

    return FLAG3D


def pack_data(data3d_dir, annots_path):
    """
    The function reads all the ground truth and prediction files. And concatenates

    :param paths_gt: all the paths corresponding to the ground truth - list of pkl files
    :param paths_prd: all the paths corresponding to the predictions - list of pkl files
    :return:
        jp_pred: jointPositions Prediction. Shape N x 24 x 3
        jp_gt: jointPositions ground truth. Shape: N x 24 x 3
        mats_pred: Global rotation matrices predictions. Shape N x 24 x 3 x 3
        mats_gt: Global rotation matrices ground truths. Shape N x 24 x 3 x 3
    """
    # all ground truth smpl parameters / joint positions / rotation matrices
    from evaluation.pw3d_eval.SMPL import SMPL

    all_params, all_jp_gts, all_jp2d_gts, all_glob_rot_gts = {}, {}, {}, {}
    seq = 0
    num_jps_pred = 0
    num_ors_pred = 0
    paths_gt = glob.glob(os.path.join(data3d_dir, '*/*.pkl'))

    smpl_model_genders = {
        'f': SMPL(center_idx=0, gender='f', model_root=os.path.join(config.model_dir, 'smpl_original')),
        'm': SMPL(center_idx=0, gender='m', model_root=os.path.join(config.model_dir, 'smpl_original'))}

    # construct the data structures -
    for path_gt in paths_gt:
        print('Processing: ', path_gt)
        video_name = os.path.basename(path_gt)
        seq = seq + 1
        # Open pkl files
        data_gt = pickle.load(open(path_gt, 'rb'), encoding='latin1')
        split = path_gt.split('/')[-2]

        genders = data_gt['genders']
        all_params[video_name], all_jp_gts[video_name], all_jp2d_gts[video_name], all_glob_rot_gts[
            video_name] = {}, [], [], []
        all_params[video_name]['split'] = split
        all_params[video_name]['genders'] = genders
        all_params[video_name]['poses'], all_params[video_name]['trans'], all_params[video_name][
            'valid_indices'] = [], [], []
        all_params[video_name]['betas'] = np.array(data_gt['betas'])
        for i in range(len(genders)):
            # Get valid frames
            # Frame with no zeros in the poses2d file and where campose_valid is True
            poses2d_gt = data_gt['poses2d']
            poses2d_gt_i = poses2d_gt[i]
            camposes_valid = data_gt['campose_valid']
            camposes_valid_i = camposes_valid[i]
            valid_indices = check_valid_inds(poses2d_gt_i, camposes_valid_i)
            all_jp2d_gts[video_name].append(poses2d_gt_i[valid_indices])

            # Get the ground truth SMPL body parameters - poses, betas and translation parameters
            pose_params = np.array(data_gt['poses'])
            pose_params = pose_params[i, valid_indices, :]
            shape_params = np.array(data_gt['betas'][i])
            shape_params = np.expand_dims(shape_params, 0)
            shape_params = shape_params[:, :10]
            shape_params = np.tile(shape_params, (pose_params.shape[0], 1))
            trans_params = np.array(data_gt['trans'])
            trans_params = trans_params[i, valid_indices, :]
            all_params[video_name]['trans'].append(trans_params)
            all_params[video_name]['valid_indices'].append(valid_indices)

            # Get the GT joint and vertex positions and the global rotation matrices
            verts_gt, jp_gt, glb_rot_mats_gt = smpl_model_genders[genders[i]].update(pose_params, shape_params,
                                                                                     trans_params)

            # Apply Camera Matrix Transformation to ground truth values
            cam_matrix = data_gt['cam_poses']
            new_cam_poses = np.transpose(cam_matrix, (0, 2, 1))
            new_cam_poses = new_cam_poses[valid_indices, :, :]

            # we don't have the joint regressor for female/male model. So we can't regress all 54 joints from the mesh of female/male model.
            jp_gt, glb_rot_mats_gt = apply_camera_transforms(jp_gt, glb_rot_mats_gt, new_cam_poses)
            root_rotation_cam_tranformed = transform_rot_representation(glb_rot_mats_gt[:, 0], input_type='mat',
                                                                        out_type='vec')
            pose_params[:, :3] = root_rotation_cam_tranformed
            all_params[video_name]['poses'].append(pose_params)
            all_jp_gts[video_name].append(jp_gt)
            all_glob_rot_gts[video_name].append(glb_rot_mats_gt)

    np.savez(annots_path, params=all_params, kp3d=all_jp_gts, glob_rot=all_glob_rot_gts, kp2d=all_jp2d_gts)


def with_ones(data):
    """
    Converts an array in 3d coordinates to 4d homogenous coordiantes
    :param data: array of shape A x B x 3
    :return return ret_arr: array of shape A x B x 4 where the extra dimension is filled with ones
    """
    ext_arr = np.ones((data.shape[0], data.shape[1], 1))
    ret_arr = np.concatenate((data, ext_arr), axis=2)
    return ret_arr


def apply_camera_transforms(joints, rotations, camera):
    """
    Applies camera transformations to joint locations and rotations matrices
    :param joints: B x 24 x 3
    :param rotations: B x 24 x 3 x 3
    :param camera: B x 4 x 4 - already transposed
    :return: joints B x 24 x 3 joints after applying camera transformations
             rotations B x 24 x 3 x 3 - rotations matrices after applying camera transformations
    """
    joints = with_ones(joints)  # B x 24 x 4
    joints = np.matmul(joints, camera)[:, :, :3]

    # multiply all rotation matrices with the camera rotation matrix
    # transpose camera coordinates back
    cam_new = np.transpose(camera[:, :3, :3], (0, 2, 1))
    cam_new = np.expand_dims(cam_new, 1)
    cam_new = np.tile(cam_new, (1, 24, 1, 1))
    # B x 24 x 3 x 3
    rotations = np.matmul(cam_new, rotations)

    return joints, rotations


def check_valid_inds(poses2d, camposes_valid):
    """
    Computes the indices where further computations are required
    :param poses2d: N x 18 x 3 array of 2d Poses
    :param camposes_valid: N x 1 array of indices where camera poses are valid
    :return: array of indices indicating frame ids in the sequence which are to be evaluated
    """

    # find all indices in the N sequences where the sum of the 18x3 array is not zero
    # N, numpy array
    poses2d_mean = np.mean(np.mean(np.abs(poses2d), axis=2), axis=1)
    poses2d_bool = poses2d_mean == 0
    poses2d_bool_inv = np.logical_not(poses2d_bool)

    # find all the indices where the camposes are valid
    camposes_valid = np.array(camposes_valid).astype('bool')

    final = np.logical_and(poses2d_bool_inv, camposes_valid)
    indices = np.array(np.where(final == True)[0])

    return indices


if __name__ == '__main__':
    ms.context.set_context(device_target='CPU')
    # dataset= FLAG3D(base_class=default_mode)(train_flag=False, split='test', mode='vibe')
    dataset = FLAG3D(base_class=default_mode)(train_flag=True)

    Test_Funcs[default_mode](dataset, with_3d=True, with_smpl=False)