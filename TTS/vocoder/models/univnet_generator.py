import numpy as np
import torch
import torch.nn.functional as F

from TTS.vocoder.layers.lvc_block import LVCBlock

LRELU_SLOPE = 0.1


class UnivnetGenerator(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        cond_channels,
        upsample_factors,
        lvc_layers_each_block,
        lvc_kernel_size,
        kpnet_hidden_channels,
        kpnet_conv_size,
        dropout,
        use_weight_norm=True,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.upsample_scale = np.prod(upsample_factors)
        self.lvc_block_nums = len(upsample_factors)

        # define first convolution
        self.first_conv = torch.nn.Conv1d(
            in_channels, hidden_channels, kernel_size=7, padding=(7 - 1) // 2, dilation=1, bias=True
        )

        # define residual blocks
        self.lvc_blocks = torch.nn.ModuleList()
        cond_hop_length = 1
        for n in range(self.lvc_block_nums):
            cond_hop_length = cond_hop_length * upsample_factors[n]
            lvcb = LVCBlock(
                in_channels=hidden_channels,
                cond_channels=cond_channels,
                upsample_ratio=upsample_factors[n],
                conv_layers=lvc_layers_each_block,
                conv_kernel_size=lvc_kernel_size,
                cond_hop_length=cond_hop_length,
                kpnet_hidden_channels=kpnet_hidden_channels,
                kpnet_conv_size=kpnet_conv_size,
                kpnet_dropout=dropout,
            )
            self.lvc_blocks += [lvcb]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    hidden_channels, out_channels, kernel_size=7, padding=(7 - 1) // 2, dilation=1, bias=True
                ),
            ]
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """
        # random noise
        x = torch.randn([c.shape[0], self.in_channels, c.shape[2]])
        x = x.to(self.first_conv.bias.device)
        x = self.first_conv(x)

        for n in range(self.lvc_block_nums):
            x = self.lvc_blocks[n](x, c)

        # apply final layers
        for f in self.last_conv_layers:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = f(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.utils.weight_norm(m)
                # print(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size, dilation=lambda x: 2 ** x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)

    def inference(self, c=None, x=None):
        """Perform inference.
        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).
        Returns:
            Tensor: Output tensor (T, out_channels)
        """
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float).to(next(self.parameters()).device)
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor).to(next(self.parameters()).device)
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        return self.forward(c).squeeze(0).transpose(1, 0)
