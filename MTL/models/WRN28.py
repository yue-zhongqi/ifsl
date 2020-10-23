import torch.nn as nn
from models.conv2d_mtl import Conv2dMtl
import torch.nn.functional as F


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class wide_basic_mtl(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic_mtl, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = Conv2dMtl(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dMtl(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                Conv2dMtl(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv3x3mtl(in_planes, out_planes, stride=1):
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class Wide_ResNet(nn.Module):
    def __init__(self, block, conv3x3proto, depth, widen_factor, dropout_rate, num_classes, remove_linear=False):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        self.final_feat_dim = 640
        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3proto(3, nStages[0])
        self.layer1 = self._wide_layer(block, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(block, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(block, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if remove_linear:
            self.linear = None
        else:
            self.linear = nn.Linear(nStages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dMtl):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        if self.linear is None:
            if feature:
                return out, None
            else:
                return out
        out1 = self.linear(out)
        if feature:
            return out, out1
        return out1


def WideRes28(num_classes=64, remove_linear=False):
    """Constructs a wideres-28-10 model without dropout.
    """
    return Wide_ResNet(wide_basic, conv3x3, 28, 10, 0, num_classes, remove_linear=remove_linear)


def WideRes28Mtl(num_classes=64, remove_linear=False):
    """Constructs a wideres-28-10 model without dropout.
    """
    return Wide_ResNet(wide_basic_mtl, conv3x3mtl, 28, 10, 0, num_classes, remove_linear=remove_linear)