from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision import models

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def vgg16_bn(pretrained=True, **kwargs):
    """ VGG 16-layer model (configuration "D") with batch normalization """
    model = models.vgg16_bn(pretrained=pretrained, **kwargs)
    return model
