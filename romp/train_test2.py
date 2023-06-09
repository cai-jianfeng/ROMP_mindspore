# _*_ coding:utf-8 _*_
"""
@Software: ROMP-master
@FileName: train_test2.py
@Date: 2023/6/7 18:43
@Author: caijianfeng
"""
import sys, os

sys.path.append(os.path.join(os.getcwd(), 'lib/'))

import mindspore as ms
from mindspore import nn, ops
from mindspore.dataset import GeneratorDataset as Dataloader
import numpy as np
from base import *
import time

np.set_printoptions(precision=2, suppress=True)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 6, 3, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Dense(1572864, 128)
        self.linear2 = nn.Dense(128, 10)

    def construct(self, x):
        # if isinstance(x, np.ndarray):
        #     x = ms.Tensor.from_numpy(x)
        x = self.conv(x)
        print(x.shape)
        x = self.relu(x)
        x = self.flatten(x)
        print(x.shape)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class Data:
    def __init__(self):
        self.data = np.random.rand(128, 3, 512, 512).astype(np.float32)
        self.label = np.random.randint(0, 10, (128,)).astype(np.int32)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__()
        self.keys = ['image', 'image_org', 'full_kp2d',
                     'person_centers', 'subject_ids', 'centermap',
                     'kp_3d', 'verts', 'params', 'valid_masks',
                     'root_trans', 'offsets', 'rot_flip',
                     'all_person_detected_mask', 'imgpath', 'data_set',
                     'camMats', 'camDists', 'img_scale']
        self.batch_size = 2
        self.loader = self._create_data_loader(train_flag=True)
        self.data = Dataloader(source=Data(),
                               column_names=['data', 'label'],
                               shuffle=True).batch(batch_size=self.batch_size,
                                                   drop_remainder=True,
                                                   num_parallel_workers=2)
        self.model = Net()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = nn.SGD(self.model.trainable_params(), 1e-2)

    def forward_fn(self, data, label):
        logits = self.model(data)
        loss = self.loss_fn(logits, label)
        return loss, logits

    # Define function of one-step training
    def train_step(self, data, label):
        (loss, _), grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss

    def train(self):
        size = self.loader.get_dataset_size()
        self.model.set_train()
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        for batch, data in enumerate(self.loader.create_tuple_iterator()):
            # data, label = data
            # data = data.numpy()
            data = data[0]
            data = data.transpose((0, 3, 1, 2))
            # data = ops.rand((self.batch_size, 3, 512, 512)).astype(ms.float32)
            label = ops.randint(0, 10, (self.batch_size,)).astype(ms.int32)
            # print(label.shape, '; ', label.dtype)
            loss = self.train_step(data, label)

            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


epochs = 3
trainer = Trainer()
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    trainer.train()
print("Done!")
