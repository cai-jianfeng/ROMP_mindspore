from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import mindspore  # import torch

from mindspore import ops, nn

import config
from config import args
from utils import print_dict

# 混合精度
# if args().model_precision == 'fp16':
#     from torch.cuda.amp import autocast

BN_MOMENTUM = 0.1
default_cfg = {'mode': 'val', 'calc_loss': False}


class Base(nn.Cell):
    def construct(self, meta_data, **cfg):
        if cfg['mode'] == 'matching_gts':
            return self.matching_forward(meta_data, **cfg)
        elif cfg['mode'] == 'parsing':
            return self.parsing_forward(meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (parsing/matching_gts)')

    def matching_forward(self, meta_data, **cfg):
        outputs = self.feed_forward(meta_data)
        print('===============feed forward succeed============')
        outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        print('===============matching forward succeed============')
        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))
        # print_dict(outputs)
        return outputs

    # @ops.stop_gradient()
    def parsing_forward(self, meta_data, **cfg):
        outputs = self.feed_forward(meta_data)
        outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)

        outputs['meta_data'] = meta_data
        # print_dict(outputs)
        # outputs = ops.stop_gradient(outputs)
        return outputs

    def feed_forward(self, meta_data):
        x = self.backbone(meta_data['image'])
        outputs = self.head_forward(x)
        return outputs

    # @ops.stop_gradient()
    def pure_forward(self, meta_data, **cfg):
        outputs = self.feed_forward(meta_data)
        # outputs = ops.stop_gradient(outputs)
        return outputs

    def head_forward(self, x):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def _build_gpu_tracker(self):
        # TODO: 搞清楚它是什么
        self.gpu_tracker = MemTracker()

    def init_weights(self):
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.get_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2dTranspose):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.get_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
