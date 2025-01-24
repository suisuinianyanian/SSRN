import torch
import torch.nn as nn
from component import rotate, Crop2d, Shift2d, ShiftConv2d
from component import SpectralTransform as ST
from fourierup import fresup
from tranformer import SSMCTB


class Generator(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 channels=32,
                 block_fk_num=2):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True), )

        bchannels = channels * 4
        bottleneck_fk = []
        for i in range(block_fk_num):
            bottleneck_fk.append(ST(bchannels, bchannels))
        self.bottleneck_fk = nn.Sequential(*bottleneck_fk)
        self.tran1 = SSMCTB(channels * 2)
        self.tran2 = SSMCTB(channels * 1)

        bottleneck_xt = []
        for i in range(2):
            bottleneck_xt.extend([
                ShiftConv2d(bchannels, bchannels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(bchannels),
                nn.ReLU(inplace=True), ])
        self.bottleneck_xt = nn.Sequential(*bottleneck_xt)
        self.bottleneck_xt_conv = nn.Sequential(
            nn.Conv2d(channels * 4 * 2, channels * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True))

        self.fourier_up1 = fresup(channels * 4)
        self.decoder1 = nn.Sequential(
            ShiftConv2d(channels * 4, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            ShiftConv2d(channels * 2, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True), )
        self.decoder1_conv = nn.Sequential(
            nn.Conv2d(channels * 2 * 2, channels * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True), )

        self.fourier_up2 = fresup(channels * 2)
        self.decoder2 = nn.Sequential(
            ShiftConv2d(channels * 2, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ShiftConv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True), )
        self.decoder2_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True), )

        self.out = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.shift = Shift2d((1, 0))
        self.crop = None

    def _crop(self, x, y):
        expect_size = x.shape[-2], x.shape[-1]
        out_size = y.shape[-2], y.shape[-1]
        crop_tuple = []
        for i in range(1, -1, -1):
            assert out_size[i] >= expect_size[i]
            diff = out_size[i] - expect_size[i]
            crop_tuple.append(diff // 2)
            crop_tuple.append(diff - diff // 2)
        return Crop2d(crop_tuple)

    def forward(self, img):
        x = self.encoder(img)
        x = self.bottleneck_fk(x)
        x = self.blind_spot_wrapper(self.bottleneck_xt, self.bottleneck_xt_conv, x)

        x = self.fourier_up1(x)
        x = self.blind_spot_wrapper(self.decoder1, self.decoder1_conv, x)
        x = self.tran1(x)[0]

        x = self.fourier_up2(x)
        x = self.blind_spot_wrapper(self.decoder2, self.decoder2_conv, x)
        x = self.tran2(x)[0]

        if self.crop is None:
            self.crop = self._crop(img, x)
        x = self.crop(x)
        x = self.out(x)
        return x

    def blind_spot_wrapper(self, net, conv, x):
        rotated = [rotate(x, rot) for rot in (90, 270)]
        x = torch.cat((rotated), dim=0)

        x = net(x)

        shifted = self.shift(x)
        rotated_batch = torch.chunk(shifted, 2, dim=0)
        aligned = [rotate(rotated, rot) for rotated, rot in zip(rotated_batch, (270, 90))]
        x = torch.cat(aligned, dim=1)

        return conv(x)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.is_available())
if __name__ == '__main__':
    x = torch.randn(1, 1, 1500, 120)
    x = x.to(device)
    net = Generator().to(device)
    print("The number of parameters of the network is: ",
          sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(net(x).shape)