import torch
import torch.nn as nn
import torch.nn.functional as F


class resNet(nn.Module):
    def __init__(self):
        super(resNet, self).__init__()
        self.reflect_padding1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(4, 32, 7, stride=1, padding=0, bias=False)
        self.ReLU = nn.ReLU()
        self.reflect_padding2 = nn.ReflectionPad2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=0, bias=False)
        self.downsamplingBlock = resBlock(64, 128, downsample=True, upsample=False)
        self.standardBlock = nn.ModuleList([resBlock(128, 128, downsample=False, upsample=False) for i in range(6)])
        self.upsamplingBlock = resBlock(128, 64, downsample=False, upsample=True)
        self.reflect_padding3 = nn.ReflectionPad2d(1)
        self.deconv = nn.ConvTranspose2d(64, 32, 5, stride=2, padding=4, output_padding=1, bias=False)
        self.reflect_padding4 = nn.ReflectionPad2d(3)
        self.conv3 = nn.Conv2d(32, 3, 7, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.reflect_padding1(x)
        out = self.conv1(out)
        out = self.ReLU(out)
        out = self.reflect_padding2(out)
        out = self.conv2(out)
        out = self.ReLU(out)
        out = self.downsamplingBlock(out)
        for i in range(6):
            out = self.standardBlock[i](out)
        out = self.upsamplingBlock(out)
        out = self.reflect_padding3(out)
        out = self.deconv(out)
        out = self.ReLU(out)
        out = self.reflect_padding4(out)
        out = self.conv3(out)
        out = self.sigmoid(out)
        return out


class resBlock(nn.Module):
    def __init__(self, inchannel, outchannel, downsample=False, upsample=False):
        super(resBlock, self).__init__()
        assert not (downsample and upsample)
        self.reflect_padding = nn.ReflectionPad2d(1)
        self.s_conv1 = nn.Conv2d(inchannel, outchannel, 3, stride=1, padding=0, bias=False)
        self.d_conv1 = nn.Conv2d(inchannel, outchannel, 3, stride=2, padding=0, bias=False)
        self.u_conv1 = nn.ConvTranspose2d(inchannel, outchannel, 3, stride=2, padding=3, output_padding=1, bias=False)     # padding unsure (2 or 3)
        self.BN = nn.modules.BatchNorm2d(outchannel)
        self.ReLU = nn.ReLU()
        self.conv2 = nn.Conv2d(outchannel, outchannel, 3, stride=1, padding=0, bias=False)
        self.d_shortcut = nn.Conv2d(inchannel, outchannel, 1, stride=2, padding=0, bias=False)
        self.u_shortcut = nn.Conv2d(inchannel, outchannel, 1, stride=1, padding=0, bias=False)
        # self.u_shortcut2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = downsample
        self.upsample = upsample

    def forward(self, x):
        out = self.reflect_padding(x)
        if (not self.downsample) and (not self.upsample):       # standard
            out = self.s_conv1(out)
        elif self.downsample and (not self.upsample):         # downsample
            out = self.d_conv1(out)
        elif (not self.downsample) and self.upsample:         # upsample
            out = self.u_conv1(out)
        out = self.BN(out)
        out = self.ReLU(out)
        out = self.reflect_padding(out)
        out = self.conv2(out)
        out = self.BN(out)
        if self.downsample and (not self.upsample):         # downsample
            x = self.d_shortcut(x)
        elif (not self.downsample) and self.upsample:         # upsample
            x = self.u_shortcut(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            # x = self.u_shortcut2(x)
        out += x
        return out
