# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Some code is from https://github.com/princeton-vl/pose-ae-train/blob/454d4ba113bbb9775d4dc259ef5e6c07c2ceed54/utils/group.py
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from munkres import Munkres
import numpy as np
import mindspore
from mindspore import nn, ops


def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp


def match_by_tag(inp, params):
    assert isinstance(params, Params), 'params should be class Params()'

    tag_k, loc_k, val_k = inp  # (1, 17, 5, 1) (1, 17, 5, 2) (1, 17, 5)
    # print(tag_k.shape)
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]))
    # print(default_.shape)
    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = i

        tags = tag_k[idx]
        joints = np.concatenate(
            (loc_k[idx], val_k[idx, :, None], tags), 1
        )
        # print(joints.shape)  # (5, 4)
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            if params.ignore_too_much \
                    and len(grouped_keys) == params.max_num_people:
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (
                        diff_normed,
                        np.zeros((num_added, num_added - num_grouped)) + 1e10
                    ),
                    axis=1
                )

            pairs = py_max_match(diff_normed)
            for row, col in pairs:
                if (
                        row < num_added
                        and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold
                ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]
    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
    # print(ans.shape)
    return ans


class Params(object):
    def __init__(self):
        self.num_joints = 17
        self.max_num_people = 5

        self.detection_threshold = 0.1
        self.tag_threshold = 1.
        self.use_detection_val = True
        self.ignore_too_much = True


class HeatmapParser(object):
    def __init__(self):
        self.params = Params()
        self.tag_per_joint = True
        NMS_KERNEL, NMS_PADDING = 5, 2
        self.map_size = 128
        self.pool = nn.MaxPool2d(kernel_size=NMS_KERNEL, stride=1, pad_mode="pad", padding=NMS_PADDING)

    def nms(self, det):
        maxm = self.pool(det)
        maxm = ops.equal(maxm, det).float()
        det = det * maxm
        return det

    def match(self, tag_k, loc_k, val_k):
        match = lambda x: match_by_tag(x, self.params)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def top_k(self, det, tag):
        # det = torch.Tensor(det, requires_grad=False)
        # tag = torch.Tensor(tag, requires_grad=False)

        det = self.nms(det)
        num_images = det.shape[0]
        num_joints = det.shape[1]
        h = det.shape[2]
        w = det.shape[3]
        det = det.view((num_images, num_joints, -1))
        val_k, ind = det.topk(self.params.max_num_people, dim=2)
        # ind.shape = val_k.shape = (1, 17, 5)
        tag = tag.view((tag.shape[0], tag.shape[1], w * h, -1))
        if not self.tag_per_joint:  # 不会进入
            tag = ops.broadcast_to(tag, (-1, self.params.num_joints, -1, -1))
        # tag.shape = (1, 33, 16384, 1)
        total_ind = ops.zeros_like(tag[:, :, :, 0]).astype(mindspore.int32)
        total_ind[:ind.shape[0], :ind.shape[1], :ind.shape[2]] = ind
        tag_k = ops.stack(
            [
                # TODO: torch 的 gather 支持 ind.shape 的每个 dim <= tag.shape 的每个 dim;
                # 但是 mindspore 的 ops.gather_elements 除了指定的 dim 外, 其余的 dim 必须保持一致
                # 解决方法: 从 ind 扩展 dim 到与 tag 一致, 然后再截断
                ops.gather_elements(tag[:, :, :, i], 2, total_ind)[:ind.shape[0], :ind.shape[1], :ind.shape[2]]
                for i in range(tag.shape[3])
            ],
            axis=3
        )
        # print(tag_k.shape)  # (1, 17, 5, 1)
        x = ind % w
        y = (ind / float(w)).astype(mindspore.int32)

        ind_k = ops.stack((x, y), axis=3)

        ans = {
            'tag_k': tag_k.numpy(),
            'loc_k': ind_k.numpy(),
            'val_k': val_k.numpy()
        }
        # print(ans['tag_k'].shape, ans['loc_k'].shape, ans['val_k'].shape)
        return ans

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        # print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id, 0:2] = (y + 0.5, x + 0.5)
        return ans

    def refine(self, det, tag, keypoints):
        """
        Given initial keypoint predictions, we identify missing joints
        :param det: numpy.ndarray of size (17, 128, 128)
        :param tag: numpy.ndarray of size (17, 128, 128) if not flip
        :param keypoints: numpy.ndarray of size (17, 4) if not flip, last dim is (x, y, det score, tag score)
        :return:
        """
        if len(tag.shape) == 3:
            # tag shape: (17, 128, 128, 1)
            tag = tag[:, :, :, None]

        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(np.int32)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        for i in range(keypoints.shape[0]):
            # score of joints i at all position
            tmp = det[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
            tmp2 = tmp - np.round(tt)

            # find maximum position
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            xx = x
            yy = y
            # detection score at maximum position
            val = tmp[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(det.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                    # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]

        return keypoints

    def parse(self, det, tag, adjust=True, refine=True, get_best=False):
        ans = self.match(**self.top_k(det, tag))
        if adjust:
            ans = self.adjust(ans, det)

        scores = [i[:, 2].mean() for i in ans[0]]

        if refine:
            ans = ans[0]
            # for every detected person
            for i in range(len(ans)):
                det_numpy = det[0].numpy()
                tag_numpy = tag[0].numpy()
                if not self.tag_per_joint:
                    tag_numpy = np.tile(
                        tag_numpy, (self.params.num_joints, 1, 1, 1)
                    )
                ans[i] = self.refine(det_numpy, tag_numpy, ans[i])
            ans = [ans]
        if len(scores) > 0:
            kp2ds = np.array(ans[0][:, :, :2])
            kp2ds = 2 * kp2ds / float(self.map_size) - 1
            return kp2ds, scores
        else:
            return np.zeros((1, self.params.num_joints, 2)), [0]

    def batch_parse(self, dets_tags, **kwargs):
        dets, tags = dets_tags[:, :self.params.num_joints], dets_tags[:, self.params.num_joints:]
        results, scores = [], []
        for det, tag in zip(dets, tags):
            kp2ds, each_scores = self.parse(det.unsqueeze(0), tag.unsqueeze(0), **kwargs)
            results.append(kp2ds)
            scores.append(each_scores)
        return results, scores


if __name__ == '__main__':
    HP = HeatmapParser()
    import cv2
    import mindspore.ops as ops

    hp = ops.rand(32, 50, 128, 128)  # torch.from_numpy(cv2.imread('test_sahg.png')[:,:,0]).unsqueeze(0).unsqueeze(0).repeat(2,25,1,1).float()
    result = HP.batch_parse(hp, get_best=False)
    # len(result) = 2  => result.type = tuple
    # len(result[0]) = 32 => result[0].type = list
    # len(result[1]) = 32 => result[1].type = list
    # result[0][0].shape = (5, 17, 2) => result[0][0].type = np.array
    # len(result[1][0]) = 5 => result[1][0].type = list
    # print(result)
