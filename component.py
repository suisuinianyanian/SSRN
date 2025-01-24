import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from format import (
    DataFormat,
    DataDim,
    DATA_FORMAT_DIM_INDEX,
    )
from math import ceil



# 旋转
def rotate(
    x: torch.Tensor, angle: int, data_format: str = DataFormat.BCHW
    ) -> torch.Tensor:
    dims = DATA_FORMAT_DIM_INDEX[data_format]
    h_dim = dims[DataDim.HEIGHT]
    w_dim = dims[DataDim.WIDTH]

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")



# 裁剪
class Crop2d(nn.Module):

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x: Tensor) -> Tensor:
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


# 平移
class Shift2d(nn.Module):

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_block(x)

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels * 2,
                                                  out_channels=out_channels // 2,
                                                  kernel_size=1, stride=1, padding=0, bias=False),
                                        nn.BatchNorm2d(out_channels // 2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=out_channels // 2,
                                                  out_channels=out_channels * 2,
                                                  kernel_size=1, stride=1, padding=0, bias=False),
                                        nn.BatchNorm2d(out_channels * 2),
                                        nn.ReLU(inplace=True))

    def forward(self, x):
        batch = x.shape[0]
        
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        # (batch, c*2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)

        # (batch, c, h, w/2+1, 2)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=(-2, -1), norm='ortho')
        
        return output



# 频谱变换
class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, split_no=2):
        super(SpectralTransform, self).__init__()
        self.split_no = split_no
        self.fu = FourierUnit(in_channels * (split_no ** 2), in_channels * (split_no ** 2))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),)

    def forward(self, x):
        n, c, h, w = x.shape
        split_no = self.split_no

        pad_h = ceil(h / split_no) * split_no - h
        pad_w = ceil(w / split_no) * split_no - w
        split_s_h = (h + pad_h) // split_no
        split_s_w = (w + pad_w) // split_no
        xs = F.pad(x, (pad_w, 0, pad_h, 0))
        
        xs = torch.cat(torch.split(
            xs, split_s_h, dim=-2), dim=1).contiguous()
        xs_c = xs.size(1)
        xs = torch.cat(torch.split(
            xs, split_s_w, dim=-1), dim=1).contiguous()
        
        xs = self.fu(xs)

        xs = torch.cat(torch.split(
            xs, xs_c, dim=1), dim=-1).contiguous()
        xs = torch.cat(torch.split(
            xs, c, dim=1), dim=-2).contiguous()

        xs = xs[:,:,pad_h:,pad_w:]
        return torch.tanh(self.conv(xs))




class ShiftSpectralTransform(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        super(ShiftSpectralTransform, self).__init__(*args, **kwargs)
        self.st = SpectralTransform(self.in_channels, self.out_channels)
        
        self.shift_size = (self.kernel_size[0] // 2, 0)
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = super(ShiftSpectralTransform, self).forward(x)
        x = self.st(x)
        x = self.crop(x)
        return x

class ShiftConv2d(nn.Conv2d):
    
    def __init__(self, *args, **kwargs):
        super(ShiftConv2d, self).__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x):
        x = self.pad(x)
        x = super(ShiftConv2d, self).forward(x)
        x = self.crop(x)
        return x

class ShiftBlock(nn.Module):

    def __init__(self, channels):
        super(ShiftBlock, self).__init__()

        self.sst = nn.Sequential(ShiftSpectralTransform(channels, channels, 3, 1, 1, bias = False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace = True),
                                 ShiftSpectralTransform(channels, channels, 3, 1, 1, bias = False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace = True),
                                 ShiftSpectralTransform(channels, channels, 3, 1, 1, bias = False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace = True),
                                 ShiftConv2d(channels, channels, 3, 1, 1, bias = False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace = True),)

    def forward(self, x):
        return x + self.sst(x)
