import torch
from mindspore import Tensor, save_checkpoint
import tensorflow as tf

def pytorch2mindspore(default_file = 'torch_resnet.pth'):
    # read pth file
    par_dict = torch.load(default_file)['state_dict']
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        params_list.append(param_dict)
    save_checkpoint(params_list,  'smpl_packed_info.ckpt')


# def convert(bin, path, ckptpath):
#     with tf.Se

pytorch2mindspore(default_file='smpl_packed_info.pth')