import torch
import torch.nn as nn
from .lay_nor import LayerNorm
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from .encoder import Encoder


class PreNorm(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.relu = nn.GELU()
        self.out_conv = nn.Conv2d(dim * 2, dim, 1, 1, bias=False)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out = torch.cat((out1, out2), dim=1)
        return self.out_conv(self.relu(out))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum)
        return scale


class BandScore(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        # print(dim)
        strs_qkv = ['q', 'k', 'v']
        for s in strs_qkv:
            self.add_module(f'to_{s}', nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
            ))
        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)
        self.ffn = PreNorm(dim)
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads
        self.chan_score = ChannelGate(dim)

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        x_norm = self.norm_in(x)
        q = self.__getattr__('to_q')(x_norm)
        k = self.__getattr__('to_k')(x_norm)
        v = self.__getattr__('to_v')(x_norm)

        q = rearrange(q, 'b (n c) h w -> b n c (h w)', n=self.num_heads)
        k = rearrange(k, 'b (n c) h w -> b n c (h w)', n=self.num_heads)
        v = rearrange(v, 'b (n c) h w -> b n c (h w)', n=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weight = rearrange(attn, 'b n c g -> b (n c) g')
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b n c (h w) -> b (n c) h w', h=h)
        out = self.norm_out(out)
        out = self.ffn(out)
        spec_score = self.chan_score(out)
        return spec_score, attn_weight


class AdaptiveMixing(nn.Module):
    def __init__(self, mix_radius=10):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.randn(1, 2 * mix_radius + 1))
        self.mix_radius = mix_radius

    def spectral_mixing(self, X, ada_mix):
        kernel_size = 2 * self.mix_radius + 1
        ada_mix = torch.softmax(ada_mix, dim=-1).unsqueeze(-1).unsqueeze(-1)
        pad = self.mix_radius
        X_padded = F.pad(X, (0, 0, 0, 0, pad, pad), mode='constant')  # [B, C+2*pad, H, W]
        X_unfold = X_padded.unfold(1, kernel_size, 1).permute(0, 1, 4, 2, 3)  # [B, C, kernel_size, H, W]
        X_mix = (X_unfold * ada_mix).sum(dim=2)
        return X_mix

    def forward(self, x, ada_mask):
        return self.spectral_mixing(x, ada_mask)


class SPDDA(nn.Module):
    def __init__(self, in_dim, max_radius=21):
        super().__init__()

        self.enc = Encoder(in_dim, in_dim)
        self.band_score = BandScore(in_dim)

        self.ada_mix = AdaptiveMixing()
        self.pi_2 = np.sqrt(2 * torch.pi)
        self.eps = 1e-8
        self.x_val = torch.linspace(-5, 5, max_radius).view(1, 1, max_radius).cuda()

    def rand_down(self, x):
        score, attn_weight = self.band_score(x)
        K = np.random.randint(x.shape[1] // 2, x.shape[1])
        _, k_min = torch.topk(score, K)

        k_min, _ = torch.sort(k_min, dim=1)
        k_index = k_min
        k_min = k_min.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        min_x = torch.gather(x, dim=1, index=k_min)

        return min_x, k_index, attn_weight

    def forward(self, x):
        fea = self.enc(x)
        rand_x, k_min, attn_weight = self.rand_down(fea)
        sigmas = (attn_weight.var(dim=[2]) + self.eps).sqrt()
        sigmas = torch.unsqueeze(sigmas, dim=-1)
        gauss = (1.0 / (sigmas * self.pi_2)) * torch.exp(-0.5 * ((self.x_val / sigmas) ** 2))
        mask = (gauss.abs() <= sigmas) * gauss
        k_min = k_min.unsqueeze(-1).expand(-1, -1, mask.size(2))
        ada_mask = torch.gather(mask, dim=1, index=k_min)
        rand_mix_x = self.ada_mix(rand_x, ada_mask)
        return rand_mix_x
