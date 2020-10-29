import numpy as np
import torch
from torch import nn

from ..layers.wavegrad import DBlock, FiLM, UBlock, Conv1d


class Wavegrad(nn.Module):
    # pylint: disable=dangerous-default-value
    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 y_conv_channels=32,
                 x_conv_channels=768,
                 dblock_out_channels=[128, 128, 256, 512],
                 ublock_out_channels=[512, 512, 256, 128, 128],
                 upsample_factors=[5, 5, 3, 2, 2],
                 upsample_dilations=[[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 4, 8],
                                     [1, 2, 4, 8], [1, 2, 4, 8]]):
        super().__init__()

        self.hop_len = np.prod(upsample_factors)
        self.noise_level = None
        self.num_steps = None
        self.beta = None
        self.alpha = None
        self.alpha_hat = None
        self.noise_level = None
        self.c1 = None
        self.c2 = None
        self.sigma = None

        # dblocks
        self.dblocks = nn.ModuleList([
            Conv1d(1, y_conv_channels, 5, padding=2),
        ])
        ic = y_conv_channels
        for oc, df in zip(dblock_out_channels, reversed(upsample_factors)):
            self.dblocks.append(DBlock(ic, oc, df))
            ic = oc

        # film
        self.film = nn.ModuleList([])
        ic = y_conv_channels
        for oc in reversed(ublock_out_channels):
            self.film.append(FiLM(ic, oc))
            ic = oc

        # ublocks
        self.ublocks = nn.ModuleList([])
        ic = x_conv_channels
        for oc, uf, ud in zip(ublock_out_channels, upsample_factors, upsample_dilations):
            self.ublocks.append(UBlock(ic, oc, uf, ud))
            ic = oc

        self.x_conv = Conv1d(in_channels, x_conv_channels, 3, padding=1)
        self.out_conv = Conv1d(oc, out_channels, 3, padding=1)

    def forward(self, x, spectrogram, noise_scale):
        downsampled = []
        for film, layer in zip(self.film, self.dblocks):
            x = layer(x)
            downsampled.append(film(x, noise_scale))

        x = self.x_conv(spectrogram)
        for layer, (film_shift, film_scale) in zip(self.ublocks,
                                                   reversed(downsampled)):
            x = layer(x, film_shift, film_scale)
        x = self.out_conv(x)
        return x

    @torch.no_grad()
    def inference(self, x):
        y_n = torch.randn(x.shape[0], 1, self.hop_len * x.shape[-1], dtype=torch.float32).to(x)
        sqrt_alpha_hat = self.noise_level.unsqueeze(1).to(x)
        for n in range(len(self.alpha) - 1, -1, -1):
            y_n = self.c1[n] * (y_n -
                        self.c2[n] * self.forward(y_n, x, sqrt_alpha_hat[n]).squeeze(1))
            if n > 0:
                z = torch.randn_like(y_n)
                y_n += self.sigma[n - 1] * z
            y_n.clamp_(-1.0, 1.0)
        return y_n


    def compute_y_n(self, y_0):
        """Compute noisy audio based on noise schedule"""
        self.noise_level = self.noise_level.to(y_0)
        if len(y_0.shape) == 3:
            y_0 = y_0.squeeze(1)
        s = torch.randint(1, self.num_steps + 1, [y_0.shape[0]])
        l_a, l_b = self.noise_level[s-1], self.noise_level[s]
        noise_scale = l_a + torch.rand(y_0.shape[0]).to(y_0) * (l_b - l_a)
        noise_scale = noise_scale.unsqueeze(1)
        noise = torch.randn_like(y_0)
        noisy_audio = noise_scale * y_0 + (1.0 - noise_scale**2)**0.5 * noise
        return noise.unsqueeze(1), noisy_audio.unsqueeze(1), noise_scale[:, 0]

    def compute_noise_level(self, num_steps, min_val, max_val):
        """Compute noise schedule parameters"""
        beta = np.linspace(min_val, max_val, num_steps)
        alpha = 1 - beta
        alpha_hat = np.cumprod(alpha)
        noise_level = np.concatenate([[1.0], alpha_hat ** 0.5], axis=0)

        self.num_steps = num_steps
        # pylint: disable=not-callable
        self.beta = torch.tensor(beta.astype(np.float32))
        self.alpha = torch.tensor(alpha.astype(np.float32))
        self.alpha_hat = torch.tensor(alpha_hat.astype(np.float32))
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

        self.c1 = 1 / self.alpha**0.5
        self.c2 = (1 - self.alpha) / (1 - self.alpha_hat)**0.5
        self.sigma = ((1.0 - self.alpha_hat[:-1]) / (1.0 - self.alpha_hat[1:]) * self.beta[1:])**0.5
