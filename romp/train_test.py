# _*_ coding:utf-8 _*_
"""
@Software: ROMP-master
@FileName: train_test.py
@Date: 2023/6/5 10:47
@Author: caijianfeng
"""
import sys, os

sys.path.append(os.path.join(os.getcwd(), 'lib/'))
print(sys.path)
# from .base import *
from base import *
# from .eval import val_result
from eval import val_result
from loss_funcs import Loss, Learnable_Loss
from collections import Iterable
import mindspore as ms
from mindspore import nn, ops

# hello world!
np.set_printoptions(precision=2, suppress=True)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 6, 3, 1)
        # self.conv2 = nn.Conv2d(6, 3, 3, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Dense(1572864, 128)
        self.linear2 = nn.Dense(128, 10)

    def construct(self, x):
        print(x.shape)
        x = self.conv(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        x = self.flatten(x)
        print(x.shape)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__()
        self.keys = ['image', 'image_org', 'full_kp2d',
                     'person_centers', 'subject_ids', 'centermap',
                     'kp_3d', 'verts', 'params', 'valid_masks',
                     'root_trans', 'offsets', 'rot_flip',
                     'all_person_detected_mask', 'imgpath', 'data_set',
                     'camMats', 'camDists', 'img_scale']
        # self._build_model_()
        self.model = Net()
        # self._build_optimizer()
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=0.01)
        self.loss = nn.CrossEntropyLoss()
        # self.set_up_val_loader()
        # self._calc_loss = Loss()
        self.loader = self._create_data_loader(train_flag=True)
        # self.merge_losses = Learnable_Loss(self.loader._get_ID_num_())  # .cuda()
        # self.merge_losses = Learnable_Loss()

    def train(self):
        # Speed-reproducibility tradeoff:
        # cuda_deterministic=False is faster but less reproducible, cuda_deterministic=True is slower but more reproducible
        self.model.set_train(True)
        for epoch in range(self.epoch):
            self.train_epoch(epoch)

    # TODO: 修改 train_step 成 mindspore 的格式 => 相当于 mindspore 文档的快速入门中的 forward_fn, 执行前向过程并放回 loss
    def train_step(self,
                   data, label):
        # 这里需要重新组装meta_data
        # print(data.shape)
        # print(self.model)
        outputs = self.model(data)
        print(outputs.shape)
        loss = self.loss(outputs, label)
        print(loss)
        return loss, outputs

    def train_epoch(self, epoch):
        # run_time, data_time, losses = [AverageMeter() for i in range(3)]
        # losses_dict = AverageMeter_Dict()
        print('===begin===')
        # print(self.loader.create_tuple_iterator().__next__())
        grad_fn = ms.value_and_grad(self.train_step, None, self.optimizer.parameters, has_aux=True)
        for iter_index, meta_data in enumerate(self.loader.create_tuple_iterator()):  # len(meta_data) = 19
            # data_time.update(time.time() - batch_start_time)
            print(len(meta_data))
            # data1 = meta_data[0]
            # data2 = meta_data[1]
            # data3 = meta_data[2]
            # data4 = meta_data[3]
            # data5 = meta_data[4]
            # data6 = meta_data[5]
            # data7 = meta_data[6]
            # data8 = meta_data[7]
            # data9 = meta_data[8]
            # data10 = meta_data[9]
            # data11 = meta_data[10]
            # data12 = meta_data[11]
            # data13 = meta_data[12]
            # data14 = meta_data[13]
            # data15 = meta_data[14]
            # data16 = meta_data[15]
            # data17 = meta_data[16]
            # data18 = meta_data[17]
            # data19 = meta_data[18]
            # ============================================
            # TODO: 修改成 mindspore 的格式
            # data = meta_data[0]  # 4, 3, 512, 512
            data = ops.rand((4, 3, 512, 512)).astype(ms.float32)
            label = ops.randint(0, 10, (4, )).astype(ms.int32)
            # data = data.transpose((0, 3, 1, 2))  # shape = (4, 3, 512, 512)
            (loss, outputs), grads = grad_fn(data, label)
            # for grad in grads:
            #     print(grad.dtype)
            # grads = (grad.astype(ms.float32) for grad in grads)
            self.optimizer(grads)


def train_(self,
           data1,
           data2,
           data3,
           data4,
           data5,
           data6,
           data7,
           data8,
           data9,
           data10,
           data11,
           data12,
           data13,
           data14,
           data15,
           data16,
           data17,
           data18,
           data19):
    # 这里需要重新组装meta_data
    print('=========================')
    meta_data = [data1,
                 data2,
                 data3,
                 data4,
                 data5,
                 data6,
                 data7,
                 data8,
                 data9,
                 data10,
                 data11,
                 data12,
                 data13,
                 data14,
                 data15,
                 data16,
                 data17,
                 data18,
                 data19]
    meta_data = {self.keys[i]: value for i, value in enumerate(meta_data)}
    for key, data in meta_data.items():
        print(key, '; ', data.shape)
    loss = 0
    outputs = 0
    return loss, outputs


def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        trainer = Trainer()
        trainer.train()


if __name__ == '__main__':
    # ms.context.set_context(device_target="CPU")
    main()
