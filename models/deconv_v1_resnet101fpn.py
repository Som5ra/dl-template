import torch
from torch.nn import *
from models.heads.deconv_v1 import DeconvV1
from models.backbones.resnet101 import ResNet101
from models.backbones.resnet_fpn import ResNetFPN

class DeconvV1_Resnet101FPN(DeconvV1):
    def __init__(self, num_classes):
        super().__init__(
            backbone=ResNetFPN(ResNet101(), outplanes=1024),
            num_classes=num_classes,
            num_upsample_blocks=5,
            upsample_block_factor=2,
            num_cls_head_blocks=2,
            cls_head_block_factor=2
        )

if __name__ == "__main__":
    from torchinfo import summary
    model = DeconvV1_Resnet101FPN(num_classes=1)
    summary(model, input_size=(1, 3, 512, 512))