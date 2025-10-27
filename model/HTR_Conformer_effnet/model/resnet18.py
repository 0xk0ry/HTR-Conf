import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockEfn(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1, drop_path=0.0, se_ratio=0.25, aa=False):
        super().__init__()
        # if stride==2 and aa=True use AA path for downsample
        self.downsample = None
        if stride != 1 or in_c != out_c:
            if stride == 2 and aa:
                self.downsample = AADownsample(in_c)
                # after AA, stride=1
                proj = nn.Conv2d(in_c, out_c, 1, bias=False)
                self.proj_bn = nn.BatchNorm2d(out_c)
            else:
                self.downsample = nn.Identity()
                proj = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
                self.proj_bn = nn.BatchNorm2d(out_c)

        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=1 if (
            stride == 2 and aa) else stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.se = SE(out_c, rd=se_ratio) if se_ratio > 0 else nn.Identity()
        self.drop = DropPath(drop_path)
        self.act2 = nn.SiLU(inplace=True)

        self.proj = proj if (stride != 1 or in_c != out_c) else None

    def forward(self, x):
        identity = x
        if isinstance(self.downsample, AADownsample):
            identity = self.downsample(identity)   # AA path
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.proj is not None and not isinstance(self.downsample, AADownsample):
            identity = self.proj_bn(self.proj(identity))
        # stochastic depth on residual
        out = identity + self.drop(out)
        return self.act2(out)


class DropPath(nn.Module):
    def __init__(self, drop=0.0): super().__init__(); self.drop = drop

    def forward(self, x):
        if not self.training or self.drop == 0:
            return x
        keep = 1.0 - self.drop
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class SE(nn.Module):
    def __init__(self, c, rd=0.25):
        super().__init__()
        r = max(1, int(c * rd))
        self.fc1 = nn.Conv2d(c, r, 1)
        self.fc2 = nn.Conv2d(r, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class AADownsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):               # AA pool first, then 1x1
        return self.bn(self.conv(self.pool(x)))


class ResNet18(nn.Module):

    def __init__(self, nb_feat=384):

        self.inplanes = nb_feat // 4
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(
            1, nb_feat // 4, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_feat // 4, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
        self.layer1 = self._make_layer(
            BasicBlockEfn, nb_feat // 4, 2, stride=(2, 1))
        self.layer2 = self._make_layer(
            BasicBlockEfn, nb_feat // 2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockEfn, nb_feat, 2, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)

        return x
