# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pywt
import pywt.data


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )  # HH  4个分解滤波器

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )  # 4个合成滤波器

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):  # 对输入图像进行二维小波分解。
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)  # dw卷积
    x = x.reshape(
        b, c, 4, h // 2, w // 2
    )  # 特征图宽高各少一般，增加了第2维，代表4个分量
    return x


def inverse_wavelet_transform(x, filters):  # 从四子带重建原始图像
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
class cubic_attention(nn.Module):
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 7,
            v_kernel_size: int = 7,
    ):
        super().__init__()
        self.h_conv = nn.Sequential(Conv(channels, channels, (1, h_kernel_size),p=(0, h_kernel_size // 2), g=channels),nn.Sigmoid())
        
        self.v_conv = nn.Sequential(Conv(channels, channels, (v_kernel_size, 1),p=(v_kernel_size // 2, 0), g=channels),nn.Sigmoid())

    def forward(self, x):
        x_ap = x
        x_h = x + x * self.h_conv(x_ap)
        x_v = x_h + x_h * self.v_conv(x_h)
        return x_v

class AdaptLocalHighFrequenceMixer1(nn.Module): # 无cuba
    def __init__(self, in_channels: int,
                 out_channels: int,
                 g: int = 1,):
        super().__init__()
        self.c = in_channels
        self.proj = nn.Sequential(
                Conv(self.c, out_channels, k=1,g=g),
                Conv(out_channels, out_channels, k=3, s=1,g=out_channels,act=False),
                Conv(out_channels, out_channels, k=1,g=g) # 这里
            )

    def forward(self, x):
        y = x.clone()
        y= self.proj(y)
        return y

class AdaptLocalHighFrequenceMixer3(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 g: int = 1,):
        super().__init__()
        hidden = out_channels // 2
        self.proj = nn.Sequential(
                Conv(in_channels, out_channels, k=1,g=g),
                Conv(out_channels, out_channels, k=3, s=1,g=out_channels,act=False),
                Conv(out_channels, out_channels, k=1,g=g) # 这里
            )
        self.local_conv = Conv(hidden, hidden, k=3, s=1,g=hidden,act=True)
        self.cubic_7 = cubic_attention(hidden)

    def forward(self, x):
        y = x.clone()
        y = self.proj(y)
        y_1,y_2 = y.chunk(2, dim=1)
        y_1= self.local_conv(y_1)
        y_2 = self.cubic_7(y_2)
        y = torch.concat([y_1,y_2], dim=1)
        return y

class WT(nn.Module):
    def __init__(self, in_channels, out_channels, wt_levels=1, wt_type='db1'):
        super(WT, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float) # 得到分解滤波器和合成滤波器
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False) # 将卷积核权重设置成分解滤波器 但是固定参数冻结权重
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
            
    def forward(self, x, invert=False):
        if not invert:
            curr_x_ll = x
            curr_shape = curr_x_ll.shape
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                    curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                    curr_x_ll = F.pad(curr_x_ll, curr_pads) # 当特征图形状不是偶数时，做了一个填充

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter) # 使用分解滤波器进行小波变换的分解
            curr_x_ll = curr_x[:,:,0,:,:] # 取出低频分量LL 这个是要返回的
            curr_x_hh = curr_x[:,:,1:4,:,:]
            return curr_x_ll, curr_x_hh # 这里改成返回低频和高频
        else:
            # 和当前高频结合 此处需要的形状： b c 4 w h
            re = inverse_wavelet_transform(x, self.iwt_filter) # 使用iwt反转回去
            return re # 这里改成返回低频和高频

class FGD(nn.Module):
    def __init__(self, in_channels, out_c):
        super(FGD, self).__init__()
        self.proj = Conv(in_channels, out_c, 1) if in_channels != out_c else nn.Identity()
        self.dwt = WT(out_c, out_c)
        
        self.conv_ll = nn.Sequential(
            Conv(out_c, out_c, k=3, s=1,g=out_c,act=True),
        )
        
        self.high_mixer = AdaptLocalHighFrequenceMixer1(out_c * 3,out_channels=out_c, g=1)
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(nn.Conv2d(out_c,out_c,1), nn.GELU())
        
        # self.learnable_thresh = torch.nn.Parameter(torch.tensor(0.02), requires_grad=True)
        # self.act = nn.GELU()

        self.proj2 = Conv(out_c * 2, out_c, 1)

    def forward(self, x):
        x = self.proj(x)
        lr, hr = self.dwt(x)
        shape_hr = hr.shape
        hr = hr.reshape(shape_hr[0], shape_hr[1] * 3, shape_hr[3], shape_hr[4])
        
        hr = self.high_mixer(hr)
      
        ll = self.conv_ll(lr)
        group = [ll, hr]
        o = self.proj2(torch.concat(group, dim=1)) # 原本的，，现在想试下加法的效果
        
        hr_avg = self.avg(hr)
        o = o + torch.mul(o,self.gate(hr_avg))
        
        # gate = self.learnable_thresh * self.act(hr)
        # o = o + torch.mul(o, gate)
        
        return o

class FBAL(nn.Module):
    def __init__(self, c1, c2, norm=None): # c1 512 c2 1024
        super(FBAL, self).__init__()
        self.in_channels = c1
        self.conv1 = Conv(c1, c2, k=1)
        self.wt = WT(c2, c2) # 这个仅保留分高低频的功能
        self.proj = nn.Sequential(
            Conv(c2, c1, k=1),
            Conv(c1, c1, k=3, g=c1, act=False),
            Conv(c1, c2, k=1),
        )

        self.high_mixer = AdaptLocalHighFrequenceMixer3(3*c2, c2,g=4)
        self.point = Conv(c2, 3*c2, k=1,g=4)
        # hr 512 2h 2w    lr: 1024 h w

    def forward(self, x) -> torch.Tensor:
        lr, hr = x
        hr = self.conv1(hr)
        _, hr_hh = self.wt(hr) # 获得低频和3个高频 b c h/2 w/2    b c 3 h/2 w/2
        
        hr_shape = hr_hh.shape
        hr_hh = hr_hh.reshape(hr_shape[0], hr_shape[1] * 3, hr_shape[3], hr_shape[4])
        short_hr = self.high_mixer(hr_hh)
        hr_hh = hr_hh + self.point(short_hr)
        
        hr_hh = hr_hh.reshape(hr_shape[0], hr_shape[1], -1, hr_shape[3], hr_shape[4])
        
        freq1 = torch.cat([lr.unsqueeze(2), hr_hh], dim=2) # 此处需要的形状： b c 4 w h
        freq1 = self.wt(freq1, invert=True) # 用high frequency 重建特征图
        
        freq1 = self.proj(freq1) + freq1

        return freq1

class FGIL(nn.Module):
    def __init__(self, c1,c2, height=2, reduction=8):
        super(FGIL, self).__init__()
        self.c1 = c1
        self.c2 = c2
        dim = c2
        self.height = height
        d = max(int(dim/reduction), 4)
        # self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        # self.proj2 = Conv(c2, c1, 1) if c1 != c2 else nn.Identity()
        self.proj = Conv(c1, c2, 1)
        self.proj2 = Conv(c2, c1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # B, C, H, W = in_feats[0].shape
        x1, x2 = x
        x1_c = self.proj(x1)
        B, C, H, W = x1_c.shape
        
        in_feats = [x1_c,x2]

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        weight = in_feats*attn
        w1 = weight[:,0,:,::]
        w2 = weight[:,1,:,::]
        out1 = x1_c + w1
        out2 = x2 + w2
        out1 = self.proj2(out1)
        out = torch.concat([out1, out2],dim=1)
        return out


