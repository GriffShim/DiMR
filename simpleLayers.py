import torch
import torch.nn as nn

class GeGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeGLU, self).__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        a = self.A(x)
        b = self.gelu(self.B(x))
        return a * b


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()

        layers = []

        layers.append(GeGLU(input_dim, hidden_dims[0]))

        current_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.GELU())
            current_dim = dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Downsample(nn.Module):
    def __init__(self, res_ind, total_res, out_channels):
        super(Downsample, self).__init__()

        kernel_size = 2 ** (total_res - res_ind)
        stride = 2 ** (total_res - res_ind)

        self.conv = nn.Conv2d(3, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

