import torch
import torch.nn as nn


class CutBasedGraph(nn.Module):
    """
    Depth-wise separable convolution uses less parameters to generate output by convolution.
    :Examples:
        >>> m = DepthwiseSeparableConv(300, 200, 5, dim=1)
        >>> input_tensor = torch.randn(32, 300, 20)
        >>> output = m(input_tensor)
    """

    def __init__(self, in_ch, out_ch, k, dim=1, relu=True):
        """
        :param in_ch: input hidden dimension size
        :param out_ch: output hidden dimension size
        :param k: kernel size
        :param dim: default 1. 1D conv or 2D conv
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.relu = relu
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch,
                                            kernel_size=k, groups=in_ch, padding=k//2)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                                            kernel_size=1, padding=0)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,
                                            kernel_size=k, groups=in_ch, padding=k//2)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                            kernel_size=1, padding=0)
        else:
            raise Exception("Incorrect dimension!")

    def forward(self, x):
        """
        :Input: (N, L_in, D)
        :Output: (N, L_out, D)
        """
        x = x.transpose(1, 2)
        if self.relu:
            out = F.relu(self.pointwise_conv(self.depthwise_conv(x)), inplace=True)
        else:
            out = self.pointwise_conv(self.depthwise_conv(x))
        return out.transpose(1, 2)  # (N, L, D)
