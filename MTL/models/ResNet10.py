import torch.nn as nn
from models.conv2d_mtl import Conv2dMtl
from models.resnet_mtl import BasicBlockMtl
from models.resnet_mtl import BasicBlock


class ResNet(nn.Module):
    def __init__(self, block, conv, layers, num_classes=1000, zero_init_residual=False, remove_linear=False):
        super(ResNet, self).__init__()
        self.Conv2d = conv
        self.inplanes = 64
        self.conv1 = self.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.final_feat_dim = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if remove_linear:
            self.fc = None
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        '''
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        '''

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # conv1x1(self.inplanes, planes * block.expansion, stride),
                self.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.fc(x)
            return x, x1
        x = self.fc(x)
        return x

'''
def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model
'''


def ResNet10MTL(**kwargs):
    model = ResNet(BasicBlockMtl, Conv2dMtl, [1, 1, 1, 1], **kwargs)
    return model


def ResNet10(**kwargs):
    model = ResNet(BasicBlock, nn.Conv2d, [1, 1, 1, 1], **kwargs)
    return model