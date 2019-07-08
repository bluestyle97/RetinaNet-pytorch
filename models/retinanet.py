import math
import torch
import torch.nn as nn
from models import fpn


class RetinaNet(nn.Module):
    def __init__(self, num_classes, backbone='fpn50', num_features=256, num_anchors=9):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        assert backbone in ['fpn18', 'fpn34', 'fpn50', 'fpn101', 'fpn152']
        self.net = getattr(fpn, backbone)(pretrained=True)
        self.cls_subnet = self._make_subnet(num_features, self.num_anchors * self.num_classes)
        self.box_subnet = self._make_subnet(num_features, self.num_anchors * 4)
        self.initialize()

    def _make_subnet(self, num_features, output_size):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(num_features, num_features, 3, 1, 1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, output_size, 3, 1, 1))
        return nn.Sequential(*layers)

    def initialize(self):
        # initialize subnet
        def initialize_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.cls_subnet.apply(initialize_layer)
        self.box_subnet.apply(initialize_layer)

        # initialize class head prior
        def initialize_prior(layer):
            pi = 0.01
            b = - math.log((1 - pi) / pi)
            nn.init.constant_(layer.bias, b)
            nn.init.normal_(layer.weight, std=0.01)
        self.cls_subnet[-1].apply(initialize_prior)

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        fms = self.net(x)
        cls_preds = []
        box_preds = []
        for fm in fms:
            cls_pred = self.cls_subnet(fm)
            box_pred = self.box_subnet(fm)
            # cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size()[0], -1, self.num_classes)
            # box_pred = box_pred.permute(0, 2, 3, 1).contiguous().view(x.size()[0], -1, 4)
            cls_preds.append(cls_pred)
            box_preds.append(box_pred)
        return cls_preds, box_preds


if __name__ == '__main__':
    net = RetinaNet(80)
    data = torch.rand([16, 3, 240, 240])
    cls_preds, box_preds = net(data)
    for cls_pred, box_pred in zip(cls_preds, box_preds):
        print(cls_pred.size())
        print(box_pred.size())