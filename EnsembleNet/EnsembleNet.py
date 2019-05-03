import torch.nn as nn
import torch


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        bottleneck_planes = int(planes/4)

        self.nl1 = norm_layer(inplanes)
        self.conv1 = conv1x1(inplanes, bottleneck_planes)

        self.nl2 = norm_layer(bottleneck_planes)
        self.conv2 = conv3x3(bottleneck_planes, bottleneck_planes, stride=stride)

        self.nl3 = norm_layer(bottleneck_planes)
        self.conv3 = conv1x1(bottleneck_planes, planes)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.nl1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.nl2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.nl3(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        bottleneck_planes = int(planes/4)

        self.nl1 = norm_layer(inplanes)
        self.conv1 = conv1x1(inplanes, bottleneck_planes)

        self.nl2 = norm_layer(bottleneck_planes)
        self.conv2 = conv3x3(bottleneck_planes, bottleneck_planes, stride=stride)

        self.nl3 = norm_layer(bottleneck_planes)
        self.conv3 = conv1x1(bottleneck_planes, planes)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.nl1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.nl2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.nl3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EnsembleNet(nn.Module):
    def __init__(self, block, layers, planes=64, num_classes=1000, norm_layer=None, use_max_pool=True):
        super(EnsembleNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        planes = [int(planes * 2 ** i) for i in range(4)]

        self.conv1 = nn.Conv2d(3, planes[0], 7, 2, padding=3)

        self.layer1 = self.layer(block, planes[0], planes[0], layers[0], 1, norm_layer)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer2a = self.layer(block, planes[0], planes[1], layers[1], 2, norm_layer)
        self.layer2b = self.layer(block, planes[0], planes[1], layers[1], 2, norm_layer)

        self.layer3a = self.layer(block, planes[1], planes[2], layers[2], 2, norm_layer)
        self.layer3b = self.layer(block, planes[1], planes[2], layers[2], 2, norm_layer)
        self.layer3c = self.layer(block, planes[1], planes[2], layers[2], 2, norm_layer)
        self.layer3d = self.layer(block, planes[1], planes[2], layers[2], 2, norm_layer)

        self.layer4a = self.layer(block, planes[2], planes[3], layers[3], 2, norm_layer)
        self.layer4b = self.layer(block, planes[2], planes[3], layers[3], 2, norm_layer)
        self.layer4c = self.layer(block, planes[2], planes[3], layers[3], 2, norm_layer)
        self.layer4d = self.layer(block, planes[2], planes[3], layers[3], 2, norm_layer)

        self.norm_and_relu_1 = self.norm_and_relu(planes[0], norm_layer)
        self.norm_and_relu_2 = self.norm_and_relu(planes[3], norm_layer)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((None, None))

        self.dropout = nn.Dropout(0.5, inplace=True)
        self.fc = nn.Linear(planes[3] * 4, num_classes)

        self.use_max_pool = use_max_pool
        self.last_planes = planes[3]

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def layer(self, block, inplanes, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None

        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def norm_and_relu(self, planes, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        return nn.Sequential(
            norm_layer(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm_and_relu_1(x)

        if self.use_max_pool:
            x = self.max_pool(x)

        x = self.layer1(x)

        a = self.layer2a(x)
        b = self.layer2b(x)

        a1 = self.layer3a(a)
        a2 = self.layer3b(a)
        b1 = self.layer3c(b)
        b2 = self.layer3d(b)

        a1 = self.layer4a(a1)
        a2 = self.layer4b(a2)
        b1 = self.layer4c(b1)
        b2 = self.layer4d(b2)

        a1 = self.norm_and_relu_2(a1)
        a2 = self.norm_and_relu_2(a2)
        b1 = self.norm_and_relu_2(b1)
        b2 = self.norm_and_relu_2(b2)

        x = torch.cat((a1, a2, b1, b2), 0)

        x = self.global_avg_pool(x)

        x = x.view((-1, self.last_planes * 4))

        x = self.dropout(x)
        x = self.fc(x)

        return x
