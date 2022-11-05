import torch.nn as nn

################################################################################
# Residual blocks for ResNetSE34V2 architecture and variants
################################################################################


class SEBasicBlock(nn.Module):
    """
    Basic block for Squeeze-and-Excitation ResNet architecture:
    input -> Conv2d -> ReLU -> BatchNorm -> Conv2D -> BatchNorm -> SE + Residual
      V____________________________________________________________________^

    Here SE refers to the squeeze-and-excitation layer, which aggregates
    information across "spatial" axes to parameterize a channel-wise gate.

    Adapted from https://tinyurl.com/pdd3p8ew
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 reduction=8):

        super(SEBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    """
    Bottleneck block for Squeeze-and-Excitation ResNet architecture. Adapted
    from https://tinyurl.com/pdd3p8ew. Not used in the default ResNetSE34V2
    architecture.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation ResNet layer. Adapted from
    https://tinyurl.com/pdd3p8ew. Aggregates global spectro-temporal information
    by average-pooling the "spatial" dimensions down to one value per channel.
    These channel averages are passed through a two-layer feedforward network
    and sigmoid activation to obtain a channel-wise gate that is applied to the
    original input via multiplication.
    """
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y
