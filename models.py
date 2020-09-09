import torch
from torch import nn as nn
import json
from torchvision.models import vgg16

from collections import OrderedDict

class CSRNet(nn.Module):

    def __init__(self, cfg_path, from_weights=False):
        super(CSRNet, self).__init__()
        self.cfg_path = cfg_path
        if not from_weights:
            self.backbone = self.parse_cfg(component='backbone', channels_in=3)
            self.load_vgg16_weights()
            self.interpolation = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            self.head = self.parse_cfg(component='head', channels_in=512)
            self.init_layers(self.head)
        else:
            self.backbone = vgg16(pretrained=False).features
            self.interpolation = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            self.head = self.parse_cfg(component='head', channels_in=512)
        
    def load_vgg16_weights(self):
        mod = vgg16(pretrained=True).features
        skip_idx = 0
        for idx, m in enumerate(self.backbone):
            if len(m.state_dict()) > 0:
                print(m.state_dict()["weight"].shape)
                try:
                    m.load_state_dict(mod[idx + skip_idx].state_dict())
                except RuntimeError:
                    skip_idx += 1
                    m.load_state_dict(mod[idx + skip_idx].state_dict())

    def forward(self, x):
        x = self.backbone(x)
        x = self.interpolation(x)
        x = self.head(x)
        return x

    def init_layers(self, module):
        for m in module:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def parse_cfg(self, component='backbone', channels_in=3):
        with open(self.cfg_path, "r") as fp:
            cfg_dict = json.load(fp)
        layers = cfg_dict[component]["layers"]
        structure = cfg_dict[component]["structure"]
        network_dict = OrderedDict()
        for layer in structure:
            layer_struct = layers[layer]
            if 'max' in layer:
                network_dict[layer] = nn.MaxPool2d(
                    kernel_size=layer_struct["kernel"], 
                    stride=layer_struct["stride"]
                )
            else:
                if layer == 'conv6' and component == 'head':
                    padding = 0
                else:
                    padding = layer_struct["dilation"]
                network_dict[layer] = nn.Conv2d(
                    channels_in, layer_struct["channels_out"],
                    layer_struct["kernel"],
                    padding=padding,
                    dilation=layer_struct["dilation"]
                )
                network_dict["ReLU" + layer[-1]] = nn.ReLU(
                    inplace=True
                )
                channels_in = layer_struct["channels_out"]
                
        return nn.Sequential(network_dict)



