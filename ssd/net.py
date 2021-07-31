import os
import torch
import torch.nn as nn

from torch.quantization import QuantStub, DeQuantStub
from dropblock import DropBlock2D
from torch.cuda.amp import autocast
from ssd.layers import *
from torch.autograd import Variable
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class SSD(nn.Module):

    def __init__(self,
                 rank,
                 base,
                 head,
                 ssd_settings,
                 inference=False,
                 int8=False,
                 onnx=False):
        super(SSD, self).__init__()

        self.inference = inference
        self.int8 = int8
        self.onnx = onnx
        self.rank = rank
        self.mobilenet = nn.ModuleList(base)
        self.loc = nn.ModuleList(head[0])
        self.cnf = nn.ModuleList(head[1])
        self.reg = nn.ModuleList(head[2])
        self.l2norm_1 = L2Norm(512, 20, torch.device(rank))
        self.n_classes = ssd_settings['n_classes']
        self.top_k = ssd_settings['top_k']
        self.min_confidence = ssd_settings['confidence_threshold']
        self.nms = ssd_settings['nms']

        if self.int8:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

        if self.inference:
            self.priors = Variable(PriorBox().apply(
                {'min_dim': ssd_settings['input_dimensions'][1:],
                 'feature_maps': [ssd_settings['feature_maps'][0]],
                 'steps': [ssd_settings['steps'][0]],
                 'size': ssd_settings['object_size']}, rank))
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()
        else:
            self.l2norm_2 = L2Norm(1024, 20, torch.device(rank))

    def forward(self, x):
        if self.int8:
            return self.forward_pass(x)
        else:
            with autocast():
                return self.forward_pass(x)

    def forward_pass(self, x):
        """Applies network layers and ops on input images x"""
        sources, loc, cnf, reg = list(), list(), list(), list()
        if self.int8:
            x = self.quant(x)

        # Add base network
        for i, layer in enumerate(self.mobilenet):
            x = layer(x)
            if i == 11:
                if self.int8:
                    sources.append(x)
                else:
                    sources.append(self.l2norm_1(x))
            if i == 14:
                if self.int8:
                    sources.append(x)
                else:
                    sources.append(self.l2norm_2(x))

        # Apply multibox head to source layers
        for (x, l, c, r) in zip(sources, self.loc, self.cnf, self.reg):
            l, c, r = l(x), c(x), r(x)
            if self.int8:
                l, c, r = self.dequant(l), self.dequant(c), self.dequant(r)
            loc.append(l.permute(0, 2, 3, 1).contiguous())
            cnf.append(c.permute(0, 2, 3, 1).contiguous())
            reg.append(r.permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        cnf = torch.cat([o.view(o.size(0), -1) for o in cnf], 1)
        reg = torch.cat([o.view(o.size(0), -1) for o in reg], 1)

        # Apply correct output layer
        if self.inference and not self.onnx:
            priors = self.priors.type(type(x.data))
            priors = priors.to(torch.device(self.rank))
            output = self.detect.apply(
                loc.view(loc.size(0), -1, 2),
                self.softmax(cnf.view(cnf.size(0), -1, self.n_classes)),
                reg.view(reg.size(0), -1, 1),
                priors,
                self.n_classes,
                self.top_k,
                self.min_confidence,
                self.nms)
        else:
            output = (
                loc.view(loc.size(0), -1, 2),
                cnf.view(cnf.size(0), -1, self.n_classes),
                reg.view(reg.size(0), -1, 1))
        return output

    def load_weights(self, file_path):
        _, ext = os.path.splitext(file_path)
        if ext == '.pkl' or '.pth':
            state_dict = torch.load(file_path, map_location=lambda s, loc: s)
            self.load_state_dict(state_dict, strict=False)
            return True
        return False


def conv_bn(inp, out, int8):
    if int8:
        act = nn.ReLU()
    else:
        act = nn.PReLU(out)

    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out),
        DropBlock2D(block_size=3, drop_prob=0.1),
        act
    )


def conv_dw(inp, out, int8):
    if int8:
        act_1 = nn.ReLU()
        act_2 = nn.ReLU()
    else:
        act_1 = nn.PReLU(inp)
        act_2 = nn.PReLU(out)

    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, bias=False,
                  groups=inp),
        nn.BatchNorm2d(inp),
        act_1,
        nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out),
        act_2
    )


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups


        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            rank,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            head,
            ssd_settings,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.rank = rank
        # self.mobilenet = nn.ModuleList(base)
        self.loc = nn.ModuleList(head[0])
        self.cnf = nn.ModuleList(head[1])
        self.reg = nn.ModuleList(head[2])
        self.l2norm_1 = L2Norm(1024, 20, torch.device(rank))	# 256 for resnet18-34 and 1024 for resnet50 
        self.l2norm_2 = L2Norm(2048, 20, torch.device(rank))	# 512 for resnet18-34 and 2048 for resnet50
        self.inplanes = 64
        self.dilation = 1
        self.n_classes = ssd_settings['n_classes']
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        """Applies network layers and ops on input images x"""

        sources, loc, cnf, reg = list(), list(), list(), list()

        # Add base network
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        sources.append(self.l2norm_1(x))
        x = self.layer4(x)
        sources.append(self.l2norm_2(x))

        # Apply multibox head to source layers
        for (x, l, c, r) in zip(sources, self.loc, self.cnf, self.reg):
            # l, c, r = l(x), c(x), r(x)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            cnf.append(c(x).permute(0, 2, 3, 1).contiguous())
            reg.append(r(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        cnf = torch.cat([o.view(o.size(0), -1) for o in cnf], 1)
        reg = torch.cat([o.view(o.size(0), -1) for o in reg], 1)

        # Apply correct output layer
        output = (
            loc.view(loc.size(0), -1, 2),
            cnf.view(cnf.size(0), -1, self.n_classes),
            reg.view(reg.size(0), -1, 1))
        return output

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        rank,
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        head,
        ssd_settings,
        **kwargs: Any
) -> ResNet:
    model = ResNet(rank, block, layers, head, ssd_settings, **kwargs)
    return model


def resnet18(rank, head, ssd_settings, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(rank, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, head, ssd_settings,
                   **kwargs)


def resnet34(rank, head, ssd_settings, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(rank, 'resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, head, ssd_settings,
                   **kwargs)


def resnet50(rank, head, ssd_settings, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(rank, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, head, ssd_settings,
                   **kwargs)


def mobile_net_v1(c, int8, inference):
    layers = [conv_bn(c, 32, int8),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(32, 64, int8),
              conv_dw(64, 128, int8),
              conv_dw(128, 128, int8),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(128, 256, int8),
              conv_dw(256, 512, int8),
              conv_dw(512, 512, int8),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(512, 512, int8),
              conv_dw(512, 512, int8),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(512, 1024, int8),
              conv_dw(1024, 1024, int8)]
    if inference:
        return layers[:-3]
    return layers


def multibox(n_classes, inference):
    loc, cnf, reg = [], [], []

    if inference:
        source_channels = [512]
    else:
        source_channels = [1024, 2048]	# [512, 1024] for mobilenetV1, [256, 512] form reset18-34 and [1024, 2048] for resnet50

    for c in source_channels:
        loc += [nn.Conv2d(c, 2, kernel_size=3, padding=1, bias=False)]
        cnf += [nn.Conv2d(c, n_classes, kernel_size=3, padding=1, bias=False)]
        reg += [nn.Conv2d(c, 1, kernel_size=3, padding=1, bias=False)]

    return (loc, cnf, reg)


def build_ssd(rank, ssd_settings, inference=False, int8=False, onnx=False):

    input_dimensions = ssd_settings['input_dimensions']

    base = mobile_net_v1(input_dimensions[0], int8, inference)
    head = multibox(ssd_settings['n_classes'], inference)

    #return SSD(rank, base, head, ssd_settings, inference, int8, onnx)
    return resnet50(rank, head, ssd_settings, pretrained=False, progress=True)
