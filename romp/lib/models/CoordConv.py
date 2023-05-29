import mindspore # import torch
from mindspore import ops, nn
# from torch import nn
import mindspore.numpy as ms_np

def get_3Dcoord_maps_halfz(size, z_base):
    range_arr = ms_np.arange(size, dtype=mindspore.float32)
    z_len = len(z_base)
    Z_map = z_base.reshape(1,z_len,1,1,1).repeat(1,1,size,size,1)
    Y_map = range_arr.reshape(1,1,size,1,1).repeat(1,z_len,1,size,1) / size * 2 -1
    X_map = range_arr.reshape(1,1,1,size,1).repeat(1,z_len,size,1,1) / size * 2 -1

    out = ops.cat([Z_map,Y_map,X_map], axis=-1)
    return out

def get_3Dcoord_maps(size=128, z_base=None):
    range_arr = ms_np.arange(size, dtype=mindspore.float32)
    if z_base is None:
        Z_map = range_arr.reshape(1,size,1,1,1).repeat(1,1,size,size,1) / size * 2 -1
    else:
        Z_map = z_base.reshape(1,size,1,1,1).repeat(1,1,size,size,1)
    Y_map = range_arr.reshape(1,1,size,1,1).repeat(1,size,1,size,1) / size * 2 -1
    X_map = range_arr.reshape(1,1,1,size,1).repeat(1,size,size,1,1) / size * 2 -1

    out = ops.cat([Z_map,Y_map,X_map], axis=-1)
    return out

'''
brought from https://github.com/GlastonburyC/coord-conv-pytorch/blob/master/coordConv.py
'''

def get_coord_maps(size=128):
    xx_ones = ops.ones([1, size], dtype=mindspore.float32)
    xx_ones = xx_ones.unsqueeze(-1)

    xx_range = ms_np.arange(size, dtype=mindspore.float32).unsqueeze(0)
    xx_range = xx_range.unsqueeze(1)

    xx_channel = ops.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)

    yy_ones = ops.ones([1, size], dtype=mindspore.float32)
    yy_ones = yy_ones.unsqueeze(1)

    yy_range = ms_np.arange(size, dtype=mindspore.float32).unsqueeze(0)
    yy_range = yy_range.unsqueeze(-1)

    yy_channel = ops.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)

    xx_channel = xx_channel.permute(0, 3, 1, 2)
    yy_channel = yy_channel.permute(0, 3, 1, 2)

    xx_channel = xx_channel.float() / (size - 1)
    yy_channel = yy_channel.float() / (size - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    out = ops.cat([xx_channel, yy_channel], axis=1)
    return out


class AddCoords(nn.Cell):
    def __init__(self, radius_channel=False):
        super(AddCoords, self).__init__()
        self.radius_channel = radius_channel

    def construct(self, in_tensor):
        """
        in_tensor: (batch_size, channels, x_dim, y_dim)
        [0,0,0,0]   [0,1,2,3]
        [1,1,1,1]   [0,1,2,3]    << (i,j)th coordinates of pixels added as separate channels
        [2,2,2,2]   [0,1,2,3]
        taken from mkocabas.
        """
        batch_size_tensor = in_tensor.shape[0]

        xx_ones = ops.ones([1, in_tensor.shape[2]], dtype=mindspore.int32)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = ms_np.arange(in_tensor.shape[2], dtype=mindspore.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = ops.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = ops.ones([1, in_tensor.shape[3]], dtype=mindspore.int32)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = ms_np.arange(in_tensor.shape[3], dtype=mindspore.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = ops.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 1, 2)
        yy_channel = yy_channel.permute(0, 3, 1, 2)

        xx_channel = xx_channel.float() / (in_tensor.shape[2] - 1)
        yy_channel = yy_channel.float() / (in_tensor.shape[3] - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        out = ops.cat([in_tensor.cuda(), xx_channel.cuda(), yy_channel.cuda()], axis=1)

        if self.radius_channel:
            radius_calc = ops.sqrt(ops.pow(xx_channel - 0.5, 2) + ops.pow(yy_channel - 0.5, 2))
            out = ops.cat([out, radius_calc], axis=1).cuda()

        return out


class CoordConv(nn.Cell):
    """ add any additional coordinate channels to the input tensor """
    def __init__(self,  *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=False)
        self.conv = nn.Conv2d(*args, **kwargs)

    def construct(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose(nn.Cell):
    """CoordConvTranspose layer for segmentation tasks."""
    def __init__(self, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=False)
        self.convT = nn.Conv2dTranspose(*args, **kwargs)

    def construct(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.convT(out)
        return out

if __name__ == '__main__':
    y = get_3Dcoord_maps(16)
    print(y,y.shape)
    y = get_coord_maps(16).repeat(1,2,1,1)
    #print(y,y.shape)
    coordconv = CoordConv(5, 64, kernel_size=3, stride=2, pad_mode='pad', padding=1, bias=False).cuda()
    x = ops.rand(2,3,16,16).cuda()
    y = coordconv(x)
    #print(y.shape)
