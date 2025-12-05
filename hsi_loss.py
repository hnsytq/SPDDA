import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map.mean()
    return 1-ret

class gradient(nn.Module):
    def __init__(self, group_size=1):
        super(gradient, self).__init__()
        x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        x_kernel = torch.FloatTensor(x_kernel).unsqueeze(0).unsqueeze(0)
        y_kernel = torch.FloatTensor(y_kernel).unsqueeze(0).unsqueeze(0)
        x_kernel = x_kernel.repeat(1, group_size, 1, 1)
        y_kernel = y_kernel.repeat(1, group_size, 1, 1)
        self.x_weight = nn.Parameter(data=x_kernel, requires_grad=False)
        self.y_weight = nn.Parameter(data=y_kernel, requires_grad=False)

    def forward(self, input):
        _, c, _, _ = input.shape
        x_grad = torch.nn.functional.conv2d(input, self.x_weight, padding=1)
        y_grad = torch.nn.functional.conv2d(input, self.y_weight, padding=1)
        gradRes = torch.mean((x_grad + y_grad).float())
        return gradRes


class SelfHSILoss(nn.Module):
    def __init__(self, weight_1, weight_2):
        super().__init__()
        self.recon = nn.MSELoss()
        self.grad = gradient().to('cuda')
        self.weight_1 = weight_1
        self.weight_2 = weight_2

    def forward(self, x, mix):
        gray_x = torch.nanmean(x, dim=1, keepdim=True)
        gray_mix = torch.nanmean(mix, dim=1, keepdim=True)
        loss_int = self.recon(gray_mix, gray_x)
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        max_mix = torch.max(mix, dim=1, keepdim=True)[0]
        loss_grad = torch.norm(self.grad(max_x) - self.grad(max_mix))
        loss_ssim = ssim(max_x, max_mix)
        gray_loss = loss_int + loss_grad + loss_ssim

        hsi_odd = mix[:, ::2, :, :]
        hsi_even = mix[:, 1::2, :, :]
        if hsi_even.shape[1] != hsi_odd.shape[1]:
            hsi_odd = hsi_odd[:, :-1, :, :]

        ssim_hsi = ssim(hsi_odd, hsi_even)
        mse_hsi = self.recon(hsi_odd, hsi_even)
        hsi_loss = ssim_hsi + mse_hsi
        
        x_gray = gray_loss.detach().item()
        x_hsi = hsi_loss.detach().item()
        lam = self.weight_1 * x_gray * (x_hsi - self.weight_2 * x_gray)
        lam = (np.exp(lam) - np.exp(-lam)) / (np.exp(lam) + np.exp(-lam)) + 1
        
        return gray_loss + hsi_loss * lam
