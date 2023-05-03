import torch
from torch.nn import *


class Conv1x1(Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv = Conv2d(num_in, num_out, kernel_size=1, bias=False)
        self.norm = BatchNorm2d(num_out)
        self.active = ReLU(True)
        self.block = Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)


class Conv3x3(Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv = Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)
        self.norm = BatchNorm2d(num_out)
        self.active = ReLU(True)
        self.block = Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)


class ResNetFPN(Module):
    def __init__(self, resnet_module, outplanes=512, interpolate_mode = 'nearest'):
        super(ResNetFPN, self).__init__()
        self.resnet_module = resnet_module
        self.outplanes = outplanes
        self.interpolate_mode = interpolate_mode

        assert outplanes % 4 == 0
        outplanes = outplanes // 4

        self.laterals = Sequential(*[Conv1x1(resnet_module.outplanes // (2 ** c), outplanes) for c in reversed(range(4))])
        self.smooths = Sequential(*[Conv3x3(outplanes * c, outplanes * c) for c in range(1, 5)])
        self.pooling = MaxPool2d(2)
        
     

    def forward(self, x):
        
        features = self.resnet_module(x)

        laterals = [l(features[f]) for f, l in enumerate(self.laterals)]

        
        map4 = laterals[3]
        map3 = laterals[2] + functional.interpolate(map4, scale_factor=2, mode=self.interpolate_mode)
        map2 = laterals[1] + functional.interpolate(map3, scale_factor=2, mode=self.interpolate_mode)
        map1 = laterals[0] + functional.interpolate(map2, scale_factor=2, mode=self.interpolate_mode)

        map1 = self.smooths[0](map1)
        map2 = self.smooths[1](torch.cat([map2, self.pooling(map1)], dim=1))
        map3 = self.smooths[2](torch.cat([map3, self.pooling(map2)], dim=1))
        map4 = self.smooths[3](torch.cat([map4, self.pooling(map3)], dim=1))

        return map4


if __name__ == "__main__":
    from torchinfo import summary
    from models.backbones.resnet101 import ResNet101
    # from models.backbones.resnet50 import ResNet50
    module = ResNetFPN(ResNet101(), outplanes=512)
    summary(module, input_size=(1, 3, 256, 256))
    