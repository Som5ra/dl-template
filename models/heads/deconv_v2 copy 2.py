import torch
from torch.nn import *


class Conv1x1(Module):
    def __init__(self, num_in, num_out, use_bias=True):
        super().__init__()
        self.conv = Conv2d(num_in, num_out, kernel_size=1, use_bias=use_bias)
        self.norm = BatchNorm2d(num_out)
        self.active = ReLU(True)
        self.block = Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)


class Conv3x3(Module):
    def __init__(self, num_in, num_out, kernel_size=3, padding=1, use_bias=True):
        super().__init__()
        self.conv = Conv2d(num_in, num_out, kernel_size=kernel_size, padding=padding, use_bias=use_bias)
        self.norm = BatchNorm2d(num_out)
        self.active = ReLU(True)
        self.block = Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)

class DynamicConv(Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride = 1, use_bias=True) -> None:
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
            Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, padding='same', use_bias=use_bias)
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
    

class Shortcut(Module):
    def __init__(self, block, shortcut = 'add') -> None:
        super(Shortcut, self).__init__()
        self.block = block
        self.shortcut = shortcut
    
    def forward(self, x):
        out = self.block(x)
        if self.shortcut == 'add':
            return out + x

# class MapsFuse(Module):
#     def __init__(self, in_channels, kernel_sizes = [3, 5]):
#         super(MapsFuse, self).__init__()
    
#         self.max_pool = AdaptiveMaxPool2d(1)
#         self.avg_pool = AdaptiveAvgPool2d(1)
#         self.attn = Sequential(
#             Flatten(),
#             Linear(in_features=in_channels * 4, out_features=in_channels),
#             ReLU(True),
#             Linear(in_features=in_channels, out_features=in_channels//2),
#             ReLU(True),
#             Linear(in_features=in_channels//2, out_features=1),
#             Sigmoid()
#         )
    
#     def forward(self, x):
#         x1, x2 = x

#         batch_size, in_planes, height, width = x1.size()

#         combined_maps = torch.concat([
#             self.max_pool(x1),
#             self.avg_pool(x1),
#             self.max_pool(x2),
#             self.avg_pool(x2),
#         ], axis=1)


#         weights = self.attn(combined_maps)
        
#         fused = torch.mul(x1.view(batch_size, -1), weights,).view(batch_size, in_planes, height, width)
#         fused += torch.mul(x2.view(batch_size, -1), 1 - weights,).view(batch_size, in_planes, height, width)
#         # fused = x1 * weights
#         # fused += x2 * (1 - weights)
#         return fused.view(batch_size, in_planes, height, width)


class SPPM(Module):
    def __init__(self, in_channels, inter_channels, out_channels, pool_sizes=[1, 2, 4]) -> None:
        super(SPPM, self).__init__()
        self.pool_sizes = pool_sizes

        self.poolings = ModuleList()
        for p in self.pool_sizes:
            self.poolings.append(Sequential(
                AdaptiveAvgPool2d(p),
                Conv1x1(in_channels, inter_channels)
            ))
        
        self.merge_conv = Sequential(
            Conv3x3(inter_channels, out_channels)
        )
    
    def forward(self, x):
        batch_size, in_planes, height, width = x.size()
        pooled = None
        for i in range(len(self.poolings)):
            p = self.poolings[i](x)
            if i == 0:
                pooled = functional.interpolate(p, size=(height, width), mode='bilinear')
            else:
                pooled += functional.interpolate(p, size=(height, width), mode='bilinear')
        
        merged = self.merge_conv(pooled)
        return merged


class FusionModule(Module):
    def __init__(self, x_ch, y_ch, out_ch, kernel_sizes = [1, 3]):
        super().__init__()

        self.upscaling = Sequential(*[
            DynamicConv( in_channels=y_ch, out_channels=y_ch * 4, kernel_sizes=kernel_sizes, use_bias=False),
            PixelShuffle(2),
            BatchNorm2d(y_ch),
            ReLU(inplace=True)
        ])

        self.conv_x = Sequential(
            # Conv1x1(x_ch, y_ch, use_bias=False)
            DynamicConv( in_channels=x_ch, out_channels=y_ch, kernel_sizes=kernel_sizes, use_bias=False),
            BatchNorm2d(y_ch),
            ReLU(inplace=True)
        )

        self.conv_xy_atten = Sequential(
            DynamicConv(in_channels=2, out_channels=2, kernel_sizes=kernel_sizes),
            BatchNorm2d(2),
            ReLU(inplace=True),

            DynamicConv(in_channels=2, out_channels=1, kernel_sizes=kernel_sizes),
            BatchNorm2d(2),
        )
        
        self.conv_out = Sequential(*[
            DynamicConv( in_channels=y_ch, out_channels=out_ch, kernel_sizes=kernel_sizes),
            BatchNorm2d( out_ch ),
            ReLU(inplace=True),

            DynamicConv( in_channels=out_ch, out_channels=out_ch, kernel_sizes=kernel_sizes),
            BatchNorm2d( out_ch ),
            ReLU(inplace=True),
        ])
       
    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x


    def prepare_y(self, x, y):
        y_up = self.upscaling(y)
        return y_up


    def avg_max_reduce_channel(self, x, use_concat=True):
        mean_value = torch.mean(x, axis=1, keepdim=True)
        max_value = torch.max(x, axis=1, keepdim=True)

        if use_concat:
            res = torch.concat([mean_value, max_value], axis=1)
        else:
            res = [mean_value, max_value]
        return res

    def fuse(self, x, y):
        atten = self.avg_max_reduce_channel([x, y])
        atten = functional.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1- atten)
        out = self.conv_out(out)
        return out


    def forward(self, x, y):
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out




class DeconvV2(Module):
    def __init__(self, resnet_module, in_channels=[256, 512, 1024, 2048], out_channels_ratio = 4, outplanes=1024, kernel_sizes = [3, 5], output_logits=True):
        super(DeconvV2, self).__init__()
        self.resnet_module = resnet_module
        self.in_channels = in_channels

        self.outplanes = outplanes
        self.output_logits = output_logits

        self.sppm = SPPM(in_channels[-1], inter_channels=in_channels[-1], out_channels=in_channels[-1]//2)
        self.fuses = ModuleList()

        for i in range(len(in_channels)):
            in_high_ch = in_channels[i+1] if i < len(in_channels) - 1 else in_channels[-1]
            in_low_ch = in_channels[i]

            self.fuses.append(FusionModule(
                x_ch = in_high_ch,
                y_ch = in_low_ch,
                out_ch = in_low_ch // 2,
                kernel_sizes=[1, 3]
            ))


        # self.fuse_laterals = ModuleList() #Sequential(*[Conv1x1(resnet_module.outplanes // (2 ** c), outplanes) for c in reversed(range(4))])
        # self.fuses = ModuleList()
        # # self.fuse_attns = ModuleList()
        # for i in range(len(in_channels)):
        #     self.fuse_laterals.append(Sequential(*[
        #         DynamicConv( in_channels=in_channels[i], out_channels=in_channels[i] // out_channels_ratio, kernel_sizes=kernel_sizes ),
        #         BatchNorm2d(in_channels[i] // out_channels_ratio),
        #         ReLU(inplace=True),
        #     ]))

        #     in_ch = in_channels[i] if i == len(in_channels) - 1 else in_channels[i+1]
        #     out_ch = in_channels[i] // 2 if i == 0 else in_channels[i-1]
        #     in_ch = in_ch // out_channels_ratio
        #     out_ch = out_ch // out_channels_ratio
            

        #     # self.fuse_attns.append(MapsFuse(in_ch//2))

        #     # print(in_ch, out_ch)
        #     self.fuses.append(Sequential(*[
        #         DynamicConv( in_channels=in_ch, out_channels=out_ch * 4, kernel_sizes=kernel_sizes),
        #         PixelShuffle(2),
        #         BatchNorm2d(out_ch),
        #         ReLU(inplace=True),

        #         Shortcut(Sequential(*[
        #             DynamicConv( in_channels=out_ch, out_channels=out_ch, kernel_sizes=kernel_sizes),
        #             BatchNorm2d( out_ch ),
        #             ReLU(inplace=True),

        #             DynamicConv( in_channels=out_ch, out_channels=out_ch, kernel_sizes=kernel_sizes),
        #             BatchNorm2d( out_ch ),
        #             ReLU(inplace=True),
        #         ])),

        #         DynamicConv( in_channels=out_ch, out_channels=out_ch, kernel_sizes=kernel_sizes),
        #         BatchNorm2d( out_ch ),
        #         ReLU(inplace=True),
        #     ]))



        # final_out_ch = (in_channels[0] // out_channels_ratio) // 2
        # self.logits = Sequential(*[
        #     DynamicConv( in_channels=final_out_ch, out_channels=final_out_ch, kernel_sizes=kernel_sizes),
        #     BatchNorm2d( final_out_ch ),
        #     ReLU(inplace=True),

        #     Shortcut(Sequential(*[
        #         DynamicConv( in_channels=final_out_ch, out_channels=final_out_ch, kernel_sizes=kernel_sizes),
        #         BatchNorm2d( final_out_ch ),
        #         ReLU(inplace=True),

        #         DynamicConv( in_channels=final_out_ch, out_channels=final_out_ch, kernel_sizes=kernel_sizes),
        #         BatchNorm2d( final_out_ch ),
        #         ReLU(inplace=True),
        #     ])),
            
        #     DynamicConv( in_channels=final_out_ch, out_channels=1, kernel_sizes=kernel_sizes),
            
        # ])

    def forward(self, x):
        
        features = self.resnet_module(x)
        for f in features:
            print("feature", f.shape)

        sppm_out = self.sppm(features[-1])
        print("sppm_out", sppm_out.shape)

        merged = []
        for i in reversed(range(len(self.in_channels))):
            if i == len(self.in_channels) - 1:
                merged.append(self.fuses[i]( sppm_out, features[i] ))
            else:
                merged.append(self.fuses[i]( merged[-1], features[i]))
            
            print("merged", merged[-1].shape)

        # upscaled = []
        # fuse_laterals = [l(features[f]) for f, l in enumerate(self.fuse_laterals)]
        # for i in reversed(range(len(self.in_channels))):
        #     if i == len(self.in_channels) - 1:
        #         upscaled.append(self.fuses[i](fuse_laterals[i]))
        #     else:
        #         upscaled.append(self.fuses[i]( torch.cat([fuse_laterals[i], upscaled[-1]], dim=1) ))
        #         # upscaled.append(self.fuses[i]( torch.cat([fuse_laterals[i],  self.fuse_attns[i]([upscaled[-1], fuse_laterals[i] ]) ], dim=1) ))
               
            
            # print("upscaled", upscaled[-1].shape)

        # #
        # merged = []
        # fuse_laterals = [l(features[f]) for f, l in enumerate(self.fuse_laterals)]
        # for i, l in enumerate(reversed(fuse_laterals)):
        #     if i == 0:
        #         merged.append(self.fuses[i](l))
        #     else:
        #         merged.append(self.fuses[i]( torch.cat([l, merged[-1]], dim=1) ))

        

        logits = upscaled[-1]
        logits = self.logits(logits)
        logits = functional.interpolate(logits, scale_factor=2, mode='bilinear')
        if self.output_logits:
            return logits
        
        logits = functional.sigmoid(logits)
        return logits

       

if __name__ == "__main__":
    from torchinfo import summary
    from models.backbones.resnet152 import ResNet152
    # from models.backbones.resnet50 import ResNet50
    module = DeconvV2(ResNet152(), outplanes=1024)
    module.cuda()
    summary(module, input_size=(1, 3, 256, 256))
    