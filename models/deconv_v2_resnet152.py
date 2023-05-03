import torch
from torch.nn import *
from models.heads.deconv_v2 import DeconvV2
from models.backbones.resnet152 import ResNet152

class DeconvV2_Resnet152(DeconvV2):
    def __init__(self, output_logits = True):
        super().__init__(
            ResNet152(),
            outplanes=1024,
            output_logits=output_logits
        )

if __name__ == "__main__":
    from torchinfo import summary
    model = DeconvV2_Resnet152(num_classes=1)
    summary(model, input_size=(1, 3, 512, 512))