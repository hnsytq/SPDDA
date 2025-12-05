import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv=default_conv, n_feat=31, kernel_size=3,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, conv=default_conv):
        super().__init__()
        n_feats, n_resblocks = 48, 6
        kernel_size = 3
        act = nn.ReLU(True)
        m_head = [conv(in_dim, n_feats, kernel_size)]
        for i in range(1, n_resblocks):
            m_head.append(ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ))

        m_tail = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(1, n_resblocks)
        ]
        m_tail.append(conv(n_feats, out_dim, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        fea = self.head(x)  # b c h w
        out = self.tail(fea)
        return out
