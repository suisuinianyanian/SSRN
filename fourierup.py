import torch
import torch.nn as nn
import torch.nn.functional as F
from component import FourierUnit
from math import ceil

# class freup_Cornerdinterpolation(nn.Module):
#     def __init__(self, channels):
#         super(freup_Cornerdinterpolation, self).__init__()
#
#         self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
#                                       nn.ReLU(inplace = True),
#                                       nn.Conv2d(channels, channels, 1, 1, 0, bias=False))
#         self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
#                                       nn.ReLU(inplace = True),
#                                       nn.Conv2d(channels, channels, 1, 1, 0, bias=False))
#
#     def forward(self, x):
#         N, C, H, W = x.shape
#
#         fft_x = torch.fft.fft2(x)
#         mag_x = torch.abs(fft_x)
#         pha_x = torch.angle(fft_x)
#
#         Mag = self.amp_fuse(mag_x)
#         Pha = self.pha_fuse(pha_x)
#
#         r = x.size(2)
#         c = x.size(3)
#
#         I_Mup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()  # 记得注释
#         I_Pup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()  # 记得注释
#
#         if r % 2 == 1:
#             ir1, ir2 = r // 2 + 1, r // 2 + 1
#         else:
#             ir1, ir2 = r // 2 + 1, r // 2
#         if c % 2 == 1:
#             ic1, ic2 = c // 2 + 1, c // 2 + 1
#         else:
#             ic1, ic2 = c // 2 + 1, c // 2
#
#         I_Mup[:, :, :ir1, :ic1] = Mag[:, :, :ir1, :ic1]
#         I_Mup[:, :, :ir1, ic2 + c:] = Mag[:, :, :ir1, ic2:]
#         I_Mup[:, :, ir2 + r:, :ic1] = Mag[:, :, ir2:, :ic1]
#         I_Mup[:, :, ir2 + r:, ic2 + c:] = Mag[:, :, ir2:, ic2:]
#
#         if r % 2 == 0:
#             I_Mup[:, :, ir2, :] = I_Mup[:, :, ir2, :] * 0.5
#             I_Mup[:, :, ir2 + r, :] = I_Mup[:, :, ir2 + r, :] * 0.5
#         if c % 2 == 0:
#             I_Mup[:, :, :, ic2] = I_Mup[:, :, :, ic2] * 0.5
#             I_Mup[:, :, :, ic2 + c] = I_Mup[:, :, :, ic2 + c] * 0.5
#
#         I_Pup[:, :, :ir1, :ic1] = Pha[:, :, :ir1, :ic1]
#         I_Pup[:, :, :ir1, ic2 + c:] = Pha[:, :, :ir1, ic2:]
#         I_Pup[:, :, ir2 + r:, :ic1] = Pha[:, :, ir2:, :ic1]
#         I_Pup[:, :, ir2 + r:, ic2 + c:] = Pha[:, :, ir2:, ic2:]
#
#         if r % 2 == 0:
#             I_Pup[:, :, ir2, :] = I_Pup[:, :, ir2, :] * 0.5
#             I_Pup[:, :, ir2 + r, :] = I_Pup[:, :, ir2 + r, :] * 0.5
#         if c % 2 == 0:
#             I_Pup[:, :, :, ic2] = I_Pup[:, :, :, ic2] * 0.5
#             I_Pup[:, :, :, ic2 + c] = I_Pup[:, :, :, ic2 + c] * 0.5
#
#         real = I_Mup * torch.cos(I_Pup)
#         imag = I_Mup * torch.sin(I_Pup)
#         out = torch.complex(real, imag)
#
#         output = torch.fft.ifft2(out)
#         output = torch.abs(output)
#
#         return output
#
# class fresup(nn.Module):
#     def __init__(self, channels=32):
#         super(fresup, self).__init__()
#         self.Fup = freup_Cornerdinterpolation(channels)
#         self.fu = FourierUnit(channels * 5, channels * 4)
#         self.lc = nn.Conv2d(channels * 5, channels * 4, 3, 1, 1, bias = False)
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels*2, channels, 3, 1, 1, bias = False),
#             nn.BatchNorm2d(channels),)
#
#     def forward(self,x):
#         xn = self.Fup(x)
#         n, c, h, w = xn.shape
#         split_no = 2
#         pad_h = ceil(h / split_no) * split_no - h
#         pad_w = ceil(w / split_no) * split_no - w
#         split_s_h = (h + pad_h) // split_no
#         split_s_w = (w + pad_w) // split_no
#         xs = F.pad(xn, (pad_w, 0, pad_h, 0))
#         xs = torch.cat(torch.split(
#             xs, split_s_h, dim=-2), dim=1).contiguous()
#         xs_c = xs.size(1)
#         xs = torch.cat(torch.split(
#             xs, split_s_w, dim=-1), dim=1).contiguous()
#
#         xs = self.fu(torch.cat([xs, x], 1))
#
#         xs = torch.cat(torch.split(
#             xs, xs_c, dim=1), dim=-1).contiguous()
#         xs = torch.cat(torch.split(
#             xs, c, dim=1), dim=-2).contiguous()
#
#         xs = xs[:,:,pad_h:,pad_w:]
#         return torch.tanh(self.conv(torch.cat([xs, xn], 1)))#torch.tanh(self.conv(xn))#torch.tanh(self.conv(torch.cat([xs, xn], 1)))
    
#############################################################
class freup_Cornerdinterpolation(nn.Module):
   def __init__(self, channels):
       super(freup_Cornerdinterpolation, self).__init__()

       self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                     nn.LeakyReLU(0.1, inplace=False),
                                     nn.Conv2d(channels, channels, 1, 1, 0, bias=False))
       self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                     nn.LeakyReLU(0.1, inplace=False),
                                     nn.Conv2d(channels, channels, 1, 1, 0, bias=False))

   def forward(self, x):
       N, C, H, W = x.shape

       fft_x = torch.fft.fft2(x)
       mag_x = torch.abs(fft_x)
       pha_x = torch.angle(fft_x)

       Mag = self.amp_fuse(mag_x)
       Pha = self.pha_fuse(pha_x)

       r = x.size(2)
       c = x.size(3)

       I_Mup = torch.zeros((N, C, 2 * H, 2 * W))#.cuda()
       I_Pup = torch.zeros((N, C, 2 * H, 2 * W))#.cuda()

       if r % 2 == 1:
           ir1, ir2 = r // 2 + 1, r // 2 + 1
       else:
           ir1, ir2 = r // 2 + 1, r // 2
       if c % 2 == 1:
           ic1, ic2 = c // 2 + 1, c // 2 + 1
       else:
           ic1, ic2 = c // 2 + 1, c // 2

       I_Mup[:, :, :ir1, :ic1] = Mag[:, :, :ir1, :ic1]
       I_Mup[:, :, :ir1, ic2 + c:] = Mag[:, :, :ir1, ic2:]
       I_Mup[:, :, ir2 + r:, :ic1] = Mag[:, :, ir2:, :ic1]
       I_Mup[:, :, ir2 + r:, ic2 + c:] = Mag[:, :, ir2:, ic2:]

       if r % 2 == 0:
           I_Mup[:, :, ir2, :] = I_Mup[:, :, ir2, :] * 0.5
           I_Mup[:, :, ir2 + r, :] = I_Mup[:, :, ir2 + r, :] * 0.5
       if c % 2 == 0:
           I_Mup[:, :, :, ic2] = I_Mup[:, :, :, ic2] * 0.5
           I_Mup[:, :, :, ic2 + c] = I_Mup[:, :, :, ic2 + c] * 0.5

       I_Pup[:, :, :ir1, :ic1] = Pha[:, :, :ir1, :ic1]
       I_Pup[:, :, :ir1, ic2 + c:] = Pha[:, :, :ir1, ic2:]
       I_Pup[:, :, ir2 + r:, :ic1] = Pha[:, :, ir2:, :ic1]
       I_Pup[:, :, ir2 + r:, ic2 + c:] = Pha[:, :, ir2:, ic2:]

       if r % 2 == 0:
           I_Pup[:, :, ir2, :] = I_Pup[:, :, ir2, :] * 0.5
           I_Pup[:, :, ir2 + r, :] = I_Pup[:, :, ir2 + r, :] * 0.5
       if c % 2 == 0:
           I_Pup[:, :, :, ic2] = I_Pup[:, :, :, ic2] * 0.5
           I_Pup[:, :, :, ic2 + c] = I_Pup[:, :, :, ic2 + c] * 0.5

       real = I_Mup * torch.cos(I_Pup)
       imag = I_Mup * torch.sin(I_Pup)
       out = torch.complex(real, imag)

       output = torch.fft.ifft2(out)
       output = torch.abs(output)

       return output

class fresup(nn.Module):
   def __init__(self, channels=32):
       super(fresup, self).__init__()
       self.Fup = freup_Cornerdinterpolation(channels)
       self.fu = FourierUnit(channels * 5, channels * 4)
       self.lc = nn.Conv2d(channels * 5, channels * 4, 3, 1, 1, bias = False)
       self.conv = nn.Sequential(
           nn.Conv2d(channels * 2, channels, 3, 1, 1, bias = False),
           nn.BatchNorm2d(channels),)

   def forward(self,x):
       xn = self.Fup(x)
       n, c, h, w = xn.shape
       split_no = 2
       pad_h = ceil(h / split_no) * split_no - h
       pad_w = ceil(w / split_no) * split_no - w
       split_s_h = (h + pad_h) // split_no
       split_s_w = (w + pad_w) // split_no
       xs = F.pad(xn, (pad_w, 0, pad_h, 0))
       xs = torch.cat(torch.split(
           xs, split_s_h, dim=-2), dim=1).contiguous()
       xs_c = xs.size(1)
       xs = torch.cat(torch.split(
           xs, split_s_w, dim=-1), dim=1).contiguous()

       xs = self.fu(torch.cat([xs, x], 1))
       xs = self.lc(torch.cat([xs, x], 1))

       xs = torch.cat(torch.split(
           xs, xs_c, dim=1), dim=-1).contiguous()
       xs = torch.cat(torch.split(
           xs, c, dim=1), dim=-2).contiguous()

       xs = xs[:,:,pad_h:,pad_w:]
       return torch.tanh(self.conv(torch.cat([xs, xn], 1)))
