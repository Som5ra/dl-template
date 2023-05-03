import torch
from torch.nn import *


class DeconvV1(Module):
    def __init__(self, backbone, num_classes=1, num_upsample_blocks = 5, upsample_block_factor = 2, num_cls_head_blocks = 2, cls_head_block_factor = 2):
        super(DeconvV1, self).__init__()
        self.backbone = backbone

        in_channel = backbone.outplanes
        self.upsample = Sequential(*[
            Sequential(
                # ConvTranspose2d( in_channels=in_channel // (upsample_block_factor ** layer_idx), out_channels=in_channel // (upsample_block_factor ** (layer_idx + 1)), kernel_size=4, stride=2, padding=1, output_padding=0, bias=False ),
                
                Conv2d( in_channels=in_channel // (2 ** layer_idx), out_channels= 2 * (in_channel // (2 ** layer_idx)), kernel_size=3, padding=1 ),
                PixelShuffle(2),
                BatchNorm2d(in_channel // (2 ** (layer_idx + 1))),

                # BatchNorm2d(in_channel // (upsample_block_factor ** (layer_idx + 1))),
                ReLU(inplace=True),
            )
            for layer_idx in range(num_upsample_blocks)
        ])
      
        # in_channel = in_channel // (upsample_block_factor ** (num_upsample_blocks))
        in_channel = in_channel // (2 ** (num_upsample_blocks))
        self.cls_head_pre = Sequential(*[
            Sequential(
                Conv2d(in_channel // (cls_head_block_factor ** (layer_idx)), in_channel // (cls_head_block_factor ** (layer_idx + 1)), kernel_size=3, padding=1),
                BatchNorm2d(in_channel // (cls_head_block_factor ** (layer_idx + 1))),
                ReLU(inplace=True),
            )
            for layer_idx in range(num_cls_head_blocks)
        ])

        in_channel = in_channel // (cls_head_block_factor ** (num_cls_head_blocks))
        self.cls_head_post = Conv2d(in_channel, num_classes, kernel_size=1, stride=1, padding=0)
        self.cls_softmax = Softmax2d()

        # self.cls_head = Sequential(
        #     Conv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
        #     BatchNorm2d(in_channel // 4),
        #     ReLU(inplace=True),
        #     Conv2d(in_channel // 4, in_channel // 8, kernel_size=3, padding=1),
        #     BatchNorm2d(in_channel // 8),
        #     ReLU(inplace=True),
        #     Conv2d(in_channel // 8, in_channel // 16, kernel_size=3, padding=1),
        #     BatchNorm2d(in_channel // 16),
        #     ReLU(inplace=True),
        #     Conv2d(in_channel // 16, num_classes, kernel_size=1, stride=1, padding=0)
        # )


    def forward(self, x):
        backbone_feats = self.backbone(x)

        backbone_feats_upscaled = self.upsample(backbone_feats)
        
        class_map = self.cls_head_pre(backbone_feats_upscaled)
        class_map = self.cls_head_post(class_map).sigmoid()       
        return class_map


if __name__ == "__main__":
    from torchinfo import summary
    from models.backbones.resnet101 import ResNet101
    from models.backbones.resnet_fpn import ResNetFPN

    backbone = ResNetFPN(ResNet101(), outplanes=1024)
    mdl = DeconvV1(backbone=backbone)
    summary(mdl, input_size=(1, 3, 256, 256))
    