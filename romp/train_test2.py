# # _*_ coding:utf-8 _*_
# """
# @Software: ROMP-master
# @FileName: train_test2.py
# @Date: 2023/6/7 18:43
# @Author: caijianfeng
# """
# import mindspore as ms
# from mindspore import nn, ops
# from mindspore.dataset import GeneratorDataset as Dataloader
# import numpy as np
#
#
# class Net(nn.Cell):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv = nn.Conv2d(3, 6, 3, 1)
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Dense(1572864, 128)
#         self.linear2 = nn.Dense(128, 10)
#
#     def construct(self, x):
#         x = self.conv(x)
#         print(x.shape)
#         x = self.relu(x)
#         x = self.flatten(x)
#         print(x.shape)
#         x = self.linear(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         return x
#
#
# class dataset:
#     def __init__(self):
#         self.data = np.random.rand(128, 3, 512, 512).astype(np.float32)
#         # self.label = np.random.randint(0, 10, (128,)).astype(np.int32)
#
#     def __getitem__(self, item):
#         return self.data[item]  #, self.label[item]
#
#     def __len__(self):
#         return self.data.shape[0]
#
#
# data = dataset()
# batch_size = 2
# data = Dataloader(data, column_names=['data'], shuffle=True)
# data = data.batch(batch_size=batch_size, drop_remainder=True)  # , num_parallel_workers=self.nw
#
# model = Net()
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = nn.SGD(model.trainable_params(), 1e-2)
#
#
# # Define forward function
# def forward_fn(data, label):
#     logits = model(data)
#     loss = loss_fn(logits, label)
#     return loss, logits
#
#
# # Get gradient function
# grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
#
#
# # Define function of one-step training
# def train_step(data, label):
#     (loss, _), grads = grad_fn(data, label)
#     optimizer(grads)
#     return loss
#
#
# def train(model, dataset):
#     size = dataset.get_dataset_size()
#     model.set_train()
#     for batch, data in enumerate(dataset.create_tuple_iterator()):
#         data = data[0]
#         label = ops.randint(0, 10, (batch_size, )).astype(ms.int32)
#         print(label.shape, '; ', label.dtype)
#         loss = train_step(data, label)
#
#         loss, current = loss.asnumpy(), batch
#         print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
#
#
# epochs = 3
# for t in range(epochs):
#     print(f"Epoch {t + 1}\n-------------------------------")
#     train(model, data)
# print("Done!")


def test(*args):
    print(args)
    for i, item in enumerate(args):
        print(item)

a = [1, 2, 3]

test(*a)