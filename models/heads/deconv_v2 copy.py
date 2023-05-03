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

class DynamicConv(Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride = 1) -> None:
        super(DynamicConv, self).__init__()

        self.atten = Sequential(*[
            AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            Linear(in_features=in_channels, out_features=in_channels//4),
            ReLU(True),
            Linear(in_features=in_channels//4, out_features=len(kernel_sizes)),
            Softmax()
        ])

        self.convs = ModuleList([
            Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, padding='same')
            for k in kernel_sizes
        ])

    def forward(self, x):
        # print("x", x.shape)
        weights = self.atten(x)
        # print("weights", weights.shape)
        conv_outs = None
        for i, c in enumerate(self.convs):
            conv_out = c(x)
            batch_size, in_planes, height, width = conv_out.size()
            conv_out_weighted = torch.mul(conv_out.view(batch_size, -1), weights[:, i:i+1],).view(batch_size, in_planes, height, width)
            if i == 0:
                conv_outs = conv_out_weighted
            else:
                conv_outs += conv_out_weighted
        return conv_outs
    

class DeconvV2(Module):
    def __init__(self, resnet_module, outplanes=1024, output_logits=True):
        super(DeconvV2, self).__init__()
        self.resnet_module = resnet_module
        self.outplanes = outplanes
        self.output_logits = output_logits

        assert outplanes % 4 == 0
        outplanes = outplanes // 4


        self.fuse_laterals = Sequential(*[Conv1x1(resnet_module.outplanes // (2 ** c), outplanes) for c in reversed(range(4))])
        self.fuses = ModuleList([
            Sequential(                
                DynamicConv( in_channels=outplanes if layer_idx == 0 else 2*outplanes, out_channels=4*outplanes, kernel_sizes=[3, 5, 7]),
                PixelShuffle(2),
                BatchNorm2d(outplanes),
                ReLU(inplace=True),

                DynamicConv( in_channels=outplanes, out_channels=outplanes, kernel_sizes=[3, 5, 7]),
                BatchNorm2d( outplanes ),
                ReLU(inplace=True),

                DynamicConv( in_channels=outplanes, out_channels=outplanes, kernel_sizes=[3, 5, 7]),
                BatchNorm2d(outplanes),
                ReLU(inplace=True),
            )
            for layer_idx in range(4)
        ])

        self.logits = Sequential(*[
            DynamicConv( in_channels=outplanes, out_channels=outplanes//4, kernel_sizes=[3, 5, 7]),
            BatchNorm2d( outplanes//4 ),
            ReLU(inplace=True),

            DynamicConv( in_channels=outplanes//4, out_channels=outplanes//8, kernel_sizes=[3, 5, 7]),
            BatchNorm2d( outplanes//8 ),
            ReLU(inplace=True),

            DynamicConv( in_channels=outplanes//8, out_channels=1, kernel_sizes=[3, 5, 7]),
            
        ])

    def forward(self, x):
        
        features = self.resnet_module(x)
        for f in features:
            print(f.shape)
        
        #
        merged = []
        fuse_laterals = [l(features[f]) for f, l in enumerate(self.fuse_laterals)]
        for i, l in enumerate(reversed(fuse_laterals)):
            if i == 0:
                merged.append(self.fuses[i](l))
            else:
                merged.append(self.fuses[i]( torch.cat([l, merged[-1]], dim=1) ))

        

        merged = merged[-1]
        logits = self.logits(merged)
        logits = functional.interpolate(logits, scale_factor=2, mode='bilinear')
        if self.output_logits:
            return logits
        
        logits = functional.sigmoid(logits)
        return logits

       

if __name__ == "__main__":
    from torchinfo import summary
    from models.backbones.resnet101 import ResNet101
    # from models.backbones.resnet50 import ResNet50
    module = DeconvV2(ResNet101(), outplanes=1024)
    module.cuda()
    summary(module, input_size=(1, 3, 256, 256))
    