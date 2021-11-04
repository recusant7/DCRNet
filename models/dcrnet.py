from torch import nn
from collections import OrderedDict
import torch


__all__ = ["dcrnet"]

def equ_conv(kernel_size,dilation):
    if dilation==1:
        return kernel_size
    else:
        return  kernel_size + (kernel_size - 1) * (dilation - 1)

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, dilation=1):

        if not isinstance(kernel_size, int):
            padding = [(equ_conv(i,dilation) - 1) // 2 for i in kernel_size]
        else:
            padding = (equ_conv(kernel_size,dilation) - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, dilation=dilation, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),

        ]))

class Channel_shuffle(nn.Module):
    def __init__(self,group):
        super(Channel_shuffle,self).__init__()
        self.groups=group

    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
                   channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used befogitre view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

class DCREncoderBlock(nn.Module):
    def __init__(self):
        super(DCREncoderBlock, self).__init__()
        self.conv1=nn.Sequential(
            ConvBN(2, 2, [3,1], dilation=1),
            nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [1,3], dilation=1),
            nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [3,1], dilation=2),
            nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [1, 3], dilation=2),
            nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [3, 1], dilation=3),
            nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [1, 3], dilation=3),
        )
        self.conv2=nn.Sequential(
            ConvBN(2, 2, 3, dilation=1)
        )
        self.prelu1 = nn.PReLU(num_parameters=4, init=0.3)
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        self.conv1x1=ConvBN(4,2,1)
        self.identity = nn.Identity()
    def forward(self, x):
        identity = self.identity(x)
        res1=self.conv1(x)
        res2=self.conv2(x)
        res=self.prelu1(torch.cat((res1,res2),dim=1))
        res=self.conv1x1(res)
        return self.prelu2(identity + res)

class DCRDecoderBlock(nn.Module):
    r""" Inverted residual with extensible width and group conv
    """
    def __init__(self, expansion):
        super(DCRDecoderBlock, self).__init__()
        width = 8 * expansion
        self.width=width
        self.path1 = nn.Sequential(OrderedDict([
            ("conv_3x3", ConvBN(2, width, 3, dilation=2)),
            ("prelu1", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_1x5", ConvBN(width, width, [3, 1], groups=4*expansion, dilation=3)),
            ("shuffle1", Channel_shuffle(4 * expansion)),
            ("prelu2", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_5x1", ConvBN(width, width, [1, 3], groups=4 * expansion, dilation=3)),
            ("shuffle1", Channel_shuffle(4 * expansion)),
            ("prelu4", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_1x1", ConvBN(width, 2, 3)),
            #("prelu5", nn.PReLU(num_parameters=2, init=0.3))

        ]))
        self.path2=nn.Sequential(
            ConvBN(2,width,[1,3]),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width,width,[5,1],groups=4*expansion),
            Channel_shuffle(4*expansion),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, width, [1, 5],groups=4*expansion),
            Channel_shuffle(4 * expansion),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, 2, [3, 1]),
        )
        self.identity = nn.Identity()
        self.prelu=nn.PReLU(num_parameters=4, init=0.3)
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        self.conv1x1 = ConvBN(4,2,1)

    def forward(self, x):
        identity = self.identity(x)
        out1 = self.path1(x)
        out2=self.path2(x)
        out=self.prelu(torch.cat((out1,out2),dim=1))
        out=self.conv1x1(out)

        return self.prelu2(identity + out)

class DCRNet(nn.Module):
    def __init__(self,
                 in_channels=2,
                 reduction=4,
                 expansion=1):
        super(DCRNet, self).__init__()

        total_size, w, h = 2048, 32, 32

        self.encoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(in_channels, 2,5)),
            ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
            ("DCREncoderBlock1", DCREncoderBlock()),



        ]))
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)

        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, in_channels, 5)),
            ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
            ("DCRDecoderBlock1", DCRDecoderBlock(expansion=expansion)),
            ("DCRDecoderBlock2", DCRDecoderBlock(expansion=expansion)),
            ("sigmoid", nn.Sigmoid())
        ]))
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, h, w = x.detach().size()

        out = self.encoder_feature(x)
        out = self.encoder_fc(out.view(n, -1))

        out = self.decoder_fc(out)
        out = self.decoder_feature(out.view(n, c, h, w))

        return out

def dcrnet(reduction=4, expansion=1):
    r""" Create an DCRNet architecture.
    """
    model = DCRNet(reduction=reduction, expansion=expansion)
    return model