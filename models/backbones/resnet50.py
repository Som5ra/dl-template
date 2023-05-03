from torch.nn import *
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50(Module):
    def __init__(self, requires_grad = True) -> None:
        super(ResNet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT).requires_grad_(requires_grad)
        self.outplanes = 2048
    

    def forward(self, x):
        size = x.size()
        # print(size)
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        return [enc1, enc2, enc3, enc4]