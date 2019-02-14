import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.maxPool = nn.MaxPool2d(2, stride=2)
        self.upConv1 = nn.ConvTranspose2d(1024, 512, 2, stride = 2)
        self.upConv2 = nn.ConvTranspose2d(512, 256, 2, stride = 2)
        self.upConv3 = nn.ConvTranspose2d(256, 128, 2, stride = 2)
        self.upConv4 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
        self.conv1x1 = nn.Conv2d(64, n_classes, 1)
        self.down1 = downStep(1, 64)
        self.down2 = downStep(64, 128)
        self.down3 = downStep(128, 256)
        self.down4 = downStep(256, 512)
        self.down5 = downStep(512, 1024)
        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64, withReLU=False)


    def forward(self, x):
        # todo
        # down = downStep(1, 64)
        downStep_1_res = self.down1(x)
        downStep_2_res = self.maxPool(downStep_1_res)
        downStep_2_res = self.down2(downStep_2_res)
        downStep_3_res = self.maxPool(downStep_2_res)
        downStep_3_res = self.down3(downStep_3_res)
        downStep_4_res = self.maxPool(downStep_3_res)
        downStep_4_res = self.down4(downStep_4_res)
        downStep_5_res = self.maxPool(downStep_4_res)
        downStep_5_res = self.down5(downStep_5_res)

        upStep_1_res = self.upConv1(downStep_5_res)
        upStep_1_res = self.up1(upStep_1_res, downStep_4_res)
        upStep_2_res = self.upConv2(upStep_1_res)
        upStep_2_res = self.up2(upStep_2_res, downStep_3_res)
        upStep_3_res = self.upConv3(upStep_2_res)
        upStep_3_res = self.up3(upStep_3_res, downStep_2_res)
        upStep_4_res = self.upConv4(upStep_3_res)
        upStep_4_res = self.up4(upStep_4_res, downStep_1_res)

        x = self.conv1x1(upStep_4_res)

        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo
        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.conv2 = nn.Conv2d(outC, outC, 3)
        self.BN = nn.modules.BatchNorm2d(outC)

    def forward(self, x):
        # todo
        x = F.relu(self.conv1(x))
        x = self.BN(x)
        x = F.relu(self.conv2(x))
        x = self.BN(x)
        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.conv2 = nn.Conv2d(outC, outC, 3)
        self.BN = nn.modules.BatchNorm2d(outC)
        self.withReLU = withReLU

    def forward(self, x, x_down):
        # todo
        th = x.size()[2]
        tw = x.size()[3]
        # n, c, tw, th = x.size()
        x_down = self.__crop(x_down, th, tw)
        x = torch.cat([x, x_down], 1)
        if self.withReLU:
            x = F.relu(self.conv1(x))
            x = self.BN(x)
            x = F.relu(self.conv2(x))
            x = self.BN(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x

    def __crop(self, variable, th, tw):
        # n, c, w, h = variable.size
        h = variable.size()[2]
        w = variable.size()[3]
        y1 = int(round((h - th) / 2.))
        x1 = int(round((w - tw) / 2.))
        return variable[:, :, y1:y1 + th, x1:x1 + tw]