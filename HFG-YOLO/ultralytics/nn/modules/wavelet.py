import pywt # pip install PyWavelets==1.8.0 使用的是pywt库，但实际上只使用了小波的定义，并没有使用pywt里的小波执行函数
import pywt.data
import torch
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1), # LL
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1), # LH
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1), # HL
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0) # HH  4个分解滤波器

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0) # 4个合成滤波器

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters): # 对输入图像进行二维小波分解。
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad) # dw卷积
    x = x.reshape(b, c, 4, h // 2, w    // 2) # 特征图宽高各少一般，增加了第2维，代表4个分量
    return x


def inverse_wavelet_transform(x, filters): # 从四子带重建原始图像
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
