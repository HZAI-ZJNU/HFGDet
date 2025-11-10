import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .conv import Conv
from .block import Bottleneck

"""https://github.com/gamepai0811/fmsa"""

__all__ = ("Transformer", "TF_Down", "TF_Up_1", "TF_Up_2", "SA_Concat")

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # (seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout = 0., is_fc = True):
        super().__init__()
        self.is_fc = is_fc
        assert dim % heads == 0, "dim must be divisible by " + str(heads)
        self.dim_head = dim // heads
        self.mlp_dim = dim * 2
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        self.pe_encoding = PositionalEncoding(dim, 20000).to(next(self.parameters()).device)
        self.ffn = nn.Sequential(
            Conv(dim, dim*2, 1),
            Conv(dim*2, dim, 1, act=False)
        ) if not is_fc else FeedForward(dim, self.mlp_dim, dropout = dropout)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = self.dim_head, dropout = dropout),
                self.ffn
            ]))

    def forward(self, x):
        #[B, C, H, W] -> [B, C, HW] -> [B, HW, C]
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous() 
        e = self.pe_encoding(x).to(device=x.device)
        x = x + e

        for attn, ff in self.layers:
            x = attn(x) + x
            if self.is_fc:
                x = ff(x) + x
            else:
                x = x.transpose(1, 2).reshape(b, c, h, w).contiguous() 
                x = ff(x) + x
                x = x.flatten(2).transpose(1, 2).contiguous()

        x = self.norm(x)
        x = x.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return x


#local attention at very begining of network - Down Sample
class TF_Down(nn.Module):
    def __init__(self, c1, c2, stride, depth = 1, heads = 8, concatenate = True):
        super().__init__()
        self.stride = stride
        self.concatenate = concatenate
        self.unfold = nn.Unfold(kernel_size=(stride,stride), padding = 0, stride=stride)
        #self.pos_embedding = nn.Parameter(torch.randn(1, c1, stride, stride))
        self.transformer = Transformer(c1, depth, heads, dropout = 0., is_fc = True)
        self.proj = nn.Sequential(Conv(c1 * 4, c1 * 8, 1), Conv(c1 * 8, c2, 1)) if concatenate else \
        nn.Sequential(Conv(c1, c1 * 2, 3, 2), Conv(c1 * 2, c1 * 4, 1), Conv(c1 * 4, c2, 1))
        

    def forward(self, x):
        B, C, H, W = x.shape
        assert  H % self.stride == 0, "H must be divisible by " + str(self.stride)
        assert  W % self.stride == 0, "W must be divisible by " + str(self.stride)

        #pick up stride*stride elements and perform local attention
        patches = self.unfold(x)  # (B, C*S*S, L)
        patches = patches.permute(0, 2, 1).contiguous()  # (B, L, C*S*S)
        patches = patches.view(B * patches.shape[1], C, self.stride, self.stride)  # (B*L, C, S, S)

        transformed_patches = self.transformer(patches)
        transformed_patches = transformed_patches.reshape(B, -1, C * self.stride * self.stride)  # (B, L, C*S*S)
        transformed_patches = transformed_patches.permute(0, 2, 1).contiguous()  # (B, C*S*S, L)
        transformed_patches = transformed_patches.reshape(B, C * self.stride * self.stride, H // self.stride, W // self.stride)
        out = self.proj(transformed_patches)

        return out


#full attention before sending data to header - Up Sample
class TF_Up_1(nn.Module):
    def __init__(self, dim_in, dim_out, depth = 2, heads = 8, scale = 80, stride = 2, dropout = 0., is_fc = True):
        super().__init__()
        self.num_patches = scale**2
        self.stride = stride
        #self.pos_embedding = nn.Parameter(torch.randn(1, dim_in, scale, scale))
        self.transformer = Transformer(dim_in, depth, heads, dropout, is_fc)
        self.ffn = nn.Sequential(
            Conv(dim_in, dim_in*2, 1),
            Conv(dim_in*2, dim_out, 1, act=False)
        ) if dim_in != dim_out else nn.Identity()


    def forward(self, x):
        x = self.transformer(x)
        x = self.ffn(x)
        
        return x #, attn


class TF_Up_2(nn.Module):

    def __init__(self, stride = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        #upsample.shape = (b, c, h*self.stride, w*self.stride)
        upsample = nn.functional.interpolate(x, scale_factor=self.stride, mode='nearest')
        
        return upsample


class SA_Concat(nn.Module):

    def __init__(self, dimension=1):
        """spatial attention and concat"""
        super().__init__()
        self.d = dimension
        self.softmax = nn.Softmax(dim = -1)
        self.weight = nn.Parameter(torch.tensor(0.1))

    def cal_spatial_attention(self, x):
        b, c, h, w = x.shape
        #(b, c, h, w) -> (h, w)
        attn = torch.sum(x, dim=(0, 1))
        #for Softmax: (h, w) -> (h * w)
        attn = attn.flatten(0)
        attn = self.softmax(attn)
        attn = attn.reshape(h, w)
        return attn

    def forward(self, x):
        """Apply spatial attention to x[1] and perform concat at the end"""
        x0 = x[0]
        x1 = x[1] + self.weight * self.cal_spatial_attention(x[0]) * x[1]

        return torch.cat([x0, x1], self.d)


