import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_SSIM_L1_LOSS(nn.Module):
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=1.0,
                 K=(0.01, 0.03),
                 alpha=0.1,
                 compensation=200.0,
                 cuda_dev=0):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        if c != 1:
            raise ValueError("Input images must have a single channel.")
        
        mux = F.conv2d(x, self.g_masks, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, padding=self.pad) - muxy

        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :]  # For single channel, we only have one layer
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs

        # loss_l1 = F.l1_loss(x, y, reduction='none')
        # gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-1, length=1), padding=self.pad).mean(1)
        
        loss_l1 = F.mse_loss(x, y, reduction='none')
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-1, length=1), padding=self.pad).mean(1)

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()
