import torch
import torch.nn as nn
import torch.nn.functional as F


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def shortcut(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            return x

        x = F.avg_pool2d(x, kernel_size=1, stride=self.stride)

        channel_difference = self.out_channels - self.in_channels
        pad_before = channel_difference // 2
        pad_after = channel_difference - pad_before

        x = F.pad(
            x,
            (0, 0, 0, 0, pad_before, pad_after),
            mode="constant",
            value=0,
        )

        return x

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class CIFAR10ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.in_channels = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(
            out_channels=16,
            num_blocks=3,
            first_stride=1,
        )

        self.layer2 = self._make_layer(
            out_channels=32,
            num_blocks=3,
            first_stride=2,
        )

        self.layer3 = self._make_layer(
            out_channels=64,
            num_blocks=3,
            first_stride=2,
        )

        self.fc = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, first_stride):
        strides = [first_stride] + [1] * (num_blocks - 1)

        blocks = []

        for stride in strides:
            blocks.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
            )

            self.in_channels = out_channels

        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = (x - self.mean) / self.std

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


def create_cifar10_resnet20(num_classes=10):
    return CIFAR10ResNet20(num_classes=num_classes)