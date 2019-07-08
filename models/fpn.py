import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride, 0, bias=False),
            nn.BatchNorm2d(planes)
        ) if stride != 1 or inplanes != planes else None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, self.expansion * planes, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.expansion * planes)
        ) if stride != 1 or inplanes != self.expansion * planes else None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        assert len(num_blocks) == 4
        self.inplanes = 64

        # bottom-up layers
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.conv6 = nn.Conv2d(512 * block.expansion, 256, 3, 2, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 2, 1)

        # lateral layers
        self.p5_1 = nn.Conv2d(512 * block.expansion, 256, 1, 1, 0)
        self.p5_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.p4_1 = nn.Conv2d(256 * block.expansion, 256, 1, 1, 0)
        self.p4_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.p3_1 = nn.Conv2d(128 * block.expansion, 256, 1, 1, 0)
        self.p3_2 = nn.Conv2d(256, 256, 3, 1, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, 3, 2, 1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu6(p6))

        # top-down
        p5 = self.p5_1(c5)
        p5_upsampled = F.interpolate(p5, size=c4.size()[-2:], mode='nearest')
        p5 = self.p5_2(p5)

        p4 = self.p4_1(c4)
        p4 += p5_upsampled
        p4_upsampled = F.interpolate(p4, size=c3.size()[-2:], mode='nearest')
        p4 = self.p4_2(p4)

        p3 = self.p3_1(c3)
        p3 += p4_upsampled
        p3 = self.p3_2(p3)

        return p3, p4, p5, p6, p7


def fpn18(pretrained=False):
    model = FPN(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'], model_dir='models/')
        model.load_state_dict(state_dict, strict=False)
    return model

def fpn34(pretrained=False):
    model = FPN(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet34'], model_dir='models/')
        model.load_state_dict(state_dict, strict=False)
    return model

def fpn50(pretrained=False):
    model = FPN(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'], model_dir='models/')
        model.load_state_dict(state_dict, strict=False)
    return model

def fpn101(pretrained=False):
    model = FPN(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet101'], model_dir='models/')
        model.load_state_dict(state_dict, strict=False)
    return model

def fpn152(pretrained=False):
    model = FPN(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet152'], model_dir='models/')
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':
    from torchsummary import summary

    fpn = fpn101(pretrained=True)
    summary(fpn, (3, 1024, 1024), device='cpu')