from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ._internally_replaced_utils import load_state_dict_from_url
from .utils import _log_api_usage_once

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


# 定义一个3x3的卷积层，可以指定输入和输出的通道数、步长、分组数和膨胀系数
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    # 使用3x3的卷积核，膨胀系数来控制 padding 的大小，这样可以保证卷积后图像的尺寸不变
    return nn.Conv2d(
        in_planes,      # 输入通道数
        out_planes,     # 输出通道数
        kernel_size=3,  # 卷积核大小
        stride=stride,  # 卷积步长
        padding=dilation, # 使用膨胀系数来控制 padding 的大小
        groups=groups,  # 分组卷积，默认为1不分组
        bias=False,     # 不使用偏置
        dilation=dilation, # 膨胀系数
    )

# 定义一个1x1的卷积层，用于调整通道数
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    # 使用1x1的卷积核，步长默认为1，不使用偏置
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 定义一个基础块（BasicBlock），这是残差网络中的一个基础构建块
class BasicBlock(nn.Module):
    # 设置扩展因子，对于基本块来说，扩展因子为1
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,    # 输入通道数
            planes: int,      # 输出通道数
            stride: int = 1,  # 步长
            downsample: Optional[nn.Module] = None, # 下采样模块，用于匹配输入输出的尺寸
            groups: int = 1,  # 分组数
            base_width: int = 64, # 基础宽度，用于计算输出通道数
            dilation: int = 1, # 膨胀系数
            norm_layer: Optional[Callable[..., nn.Module]] = None, # 归一化层，默认为BatchNorm2d
    ) -> None:
        super().__init__()
        # 如果没有指定归一化层，则使用BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # BasicBlock只支持groups=1和base_width=64
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # BasicBlock不支持膨胀系数大于1
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # 定义第一个卷积层，步长和膨胀系数会影响输出尺寸
        self.conv1 = conv3x3(inplanes, planes, stride)
        # 定义归一化层
        self.bn1 = norm_layer(planes)
        # 定义ReLU激活函数，inplace=True表示直接修改输入数据，节省内存
        self.relu = nn.ReLU(inplace=True)
        # 定义第二个卷积层，不改变输出尺寸
        self.conv2 = conv3x3(planes, planes)
        # 定义第二个归一化层
        self.bn2 = norm_layer(planes)
        # 下采样模块，用于匹配输入输出的尺寸
        self.downsample = downsample
        # 步长，用于后续计算
        self.stride = stride

    # 前向传播过程
    def forward(self, x: Tensor) -> Tensor:
        # 保存输入数据，用于后面的残差连接
        identity = x

        # 通过第一个卷积层、归一化层和ReLU激活函数
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 通过第二个卷积层、归一化层
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在下采样模块，对输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 将下采样的输入与卷积块的输出相加，实现残差连接
        out += identity
        # 通过ReLU激活函数
        out = self.relu(out)

        # 返回最终的输出
        return out

# 定义Bottleneck模块
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    # 设置扩展因子，对于Bottleneck来说，扩展因子为4
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,    # 输入通道数
            planes: int,      # 输出通道数，这里的输出通道数是指中间层的通道数，最终输出通道数是planes * expansion
            stride: int = 1,  # 步长
            downsample: Optional[nn.Module] = None, # 下采样模块，用于匹配输入输出的尺寸
            groups: int = 1,  # 分组数
            base_width: int = 64, # 基础宽度，用于计算中间层的通道数
            dilation: int = 1, # 膨胀系数
            norm_layer: Optional[Callable[..., nn.Module]] = None, # 归一化层，默认为BatchNorm2d
    ) -> None:
        super().__init__()
        # 如果没有指定归一化层，则使用BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 计算中间层的通道数，根据基础宽度和分组数进行调整
        width = int(planes * (base_width / 64.0)) * groups
        # 定义第一个1x1卷积层，用于降维
        self.conv1 = conv1x1(inplanes, width)
        # 定义第一个归一化层
        self.bn1 = norm_layer(width)
        # 定义第二个3x3卷积层，步长和膨胀系数会影响输出尺寸
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # 定义第二个归一化层
        self.bn2 = norm_layer(width)
        # 定义第三个1x1卷积层，用于升维
        self.conv3 = conv1x1(width, planes * self.expansion)
        # 定义第三个归一化层
        self.bn3 = norm_layer(planes * self.expansion)
        # 定义ReLU激活函数，inplace=True表示直接修改输入数据，节省内存
        self.relu = nn.ReLU(inplace=True)
        # 下采样模块，用于匹配输入输出的尺寸
        self.downsample = downsample
        # 步长，用于后续计算
        self.stride = stride

    # 前向传播过程
    def forward(self, x: Tensor) -> Tensor:
        # 保存输入数据，用于后面的残差连接
        identity = x

        # 通过第一个卷积层、归一化层和ReLU激活函数
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 通过第二个卷积层、归一化层和ReLU激活函数
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 通过第三个卷积层和归一化层
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果存在下采样模块，对输入进行下采样
        # 这是为了确保在添加残差之前，输入和输出的维度匹配
        if self.downsample is not None:
            identity = self.downsample(x)

        # 将下采样的输入（或原始输入）与卷积块的输出相加，实现残差连接
        out += identity

        # 最后，通过ReLU激活函数输出结果
        out = self.relu(out)

        # 返回最终的输出
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],  # 残差块类型，可以是BasicBlock或Bottleneck
            layers: List[int],  # 每个残差层中残差块的重复次数
            num_classes: int = 1000,  # 输出分类数，默认为1000
            zero_init_residual: bool = False,  # 是否将残差分支的最后一个BN层初始化为0
            groups: int = 1,  # 卷积层的组数
            width_per_group: int = 64,  # 每组的宽度
            replace_stride_with_dilation: Optional[List[bool]] = None,  # 是否用空洞卷积替换stride
            norm_layer: Optional[Callable[..., nn.Module]] = None,  # 归一化层，默认为BatchNorm2d
    ) -> None:
        super().__init__()  # 初始化父类nn.Module
        _log_api_usage_once(self)  # 用于记录API使用情况，内部函数
        if norm_layer is None:  # 如果没有指定norm_layer，则使用BatchNorm2d
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer  # 保存归一化层

        self.inplanes = 64  # 初始通道数
        self.dilation = 1  # 空洞卷积的膨胀率
        if replace_stride_with_dilation is None:  # 如果没有指定replace_stride_with_dilation，则初始化为全False
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:  # 如果replace_stride_with_dilation长度不是3，则抛出异常
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups  # 组数
        self.base_width = width_per_group  # 每组宽度
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)  # 定义第一个归一化层
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 定义最大池化层
        # 定义四个残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 定义自适应平均池化层
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 定义全连接层

        # 初始化卷积层和归一化层的权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        # 如果zero_init_residual参数为True，则循环遍历模型中的所有模块，并将Bottleneck残差块中的第三个批量归一化（Batch Normalization, BN）层的权重初始化为0，
        # 以及将BasicBlock残差块中的第二个BN层的权重初始化为0。这样做是为了遵循原始的ResNet论文中的建议，以确保残差块在身份映射（identity mapping）时的路径上不会影响梯度流。
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    # _make_layer函数用于创建一个残差层，它由多个残差块组成。
    # 入参：残差块类型（block），输出通道数（planes），残差块的数量（blocks），步长（stride），是否使用空洞卷积（dilate）。
    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],  # 残差块类型，可以是BasicBlock或Bottleneck
            planes: int,  # 该层残差块的输出通道数
            blocks: int,  # 该层中残差块的数量
            stride: int = 1,  # 第一个残差块的步长，用于下采样
            dilate: bool = False,  # 是否使用空洞卷积（dilated convolution）
    ) -> nn.Sequential:
        norm_layer = self._norm_layer  # 获取归一化层，通常是BatchNorm2d
        downsample = None  # 用于下采样的模块，默认为None
        previous_dilation = self.dilation  # 保存当前的dilation，用于后续的残差块
        if dilate:  # 如果dilate为True，则更新dilation
            self.dilation *= stride
            stride = 1
        # 判断是否需要下采样，如果stride不为1或输入通道数不等于输出通道数的扩张，则需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(  # 创建下采样模块
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 1x1卷积，用于调整通道数和下采样
                norm_layer(planes * block.expansion),  # 对下采样的结果进行归一化
            )

        layers = []  # 创建一个列表，用于存储当前层的所有残差块
        # 添加第一个残差块，可能包含下采样
        layers.append(
            block(  # 创建残差块实例
                self.inplanes,  # 输入通道数
                planes,  # 输出通道数
                stride,  # 步长
                downsample,  # 下采样模块
                self.groups,  # ParallelGroup卷积的组数
                self.base_width,  # 每组的宽度
                previous_dilation,  # 之前的dilation
                norm_layer  # 归一化层
            )
        )
        # 更新输入通道数，为下一个残差块做准备
        self.inplanes = planes * block.expansion
        # 添加剩余的残差块，不包含下采样
        for _ in range(1, blocks):
            layers.append(
                block(  # 创建残差块实例
                    self.inplanes,  # 输入通道数
                    planes,  # 输出通道数
                    groups=self.groups,  # ParallelGroup卷积的组数
                    base_width=self.base_width,  # 每组的宽度
                    dilation=self.dilation,  # 当前的dilation
                    norm_layer=norm_layer  # 归一化层
                )
            )

        # 将所有残差块组合成一个nn.Sequential模块，方便前向传播
        return nn.Sequential(*layers)

    # _forward_impl方法定义了模型的前向传播过程。
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        # 输入数据x首先通过第一个卷积层、批量归一化层、ReLU激活函数和最大池化层。
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 然后，数据通过四个残差层（layer1到layer4）。
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 最后，通过全局平均池化层、展平操作和全连接层得到最终的输出。
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # forward方法简单地调用了_forward_impl方法，这是为了提供一种一致的方式来访问模型的前向传播逻辑，同时允许子类重写前向传播过程。
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any,
) -> ResNet:
    # 第一步调用ResNet生成模型
    model = ResNet(block, layers, **kwargs)
    # 第二步判断是否有预训练，若有则调用load_state_dict_from_url函数：
    # load_state_dict_from_url 函数是PyTorch框架的一部分，用于从互联网上加载预训练模型的权重。
    # load_state_dict_from_url函数接受预训练权重的URL地址和一个模型实例，它会从给定的URL下载权重，并将其加载到模型中。
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


# 下面是resnet不同层数的几个版本，都是调用_resnet进行实现
def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)