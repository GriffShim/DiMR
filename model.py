import torch
import torch.nn as nn
from .blocks import ConvNeXt, TransformerBlock
from .simpleLayers import Downsample, Upsample
from .attention import Patchify


class DiMR(nn.Module):
    def __init__(self, hidden_sizes, hidden_layers):
        super(DiMR, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.hidden_layers = hidden_layers
        patch_size = 4
        attn_chan = hidden_sizes[0] // (patch_size ** 2)
        self.downsample1 = Downsample(1, 3, attn_chan)
        self.downsample2 = Downsample(2, 3, 3)
        self.downsample3 = Downsample(3, 3, 3)

        self.upsample1 = Upsample(attn_chan, 3, 2)
        self.upsample2 = Upsample(3, 3, 2)

        self.conv1 = nn.Conv2d(attn_chan, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

        embed_dim = (patch_size ** 2) * attn_chan
        self.patchy = Patchify(patch_size, 16, embed_dim=embed_dim)
        self.transformer_blocks = nn.ModuleList([
                                                TransformerBlock(input_dim=(int((16 / patch_size) ** 2), embed_dim), num_heads=4) 
                                                for _ in range(hidden_layers[0])
                                            ])

        self.convnext_blocks1 = nn.ModuleList([ConvNeXt(dim=3, hidden_dim=hidden_sizes[1], input_size=(3,32,32)) for _ in range(hidden_layers[1])])
        self.convnext_blocks2 = nn.ModuleList([ConvNeXt(dim=3, hidden_dim=hidden_sizes[2], input_size=(3,64,64)) for _ in range(hidden_layers[2])])

    def forward(self, x, t):
        out1 = self.downsample1(x)
        out1 = self.patchy(out1)
        for block in self.transformer_blocks:
            out1 = block(out1, t)
        out1 = self.patchy.unpatchify(out1)
        out1_upsample = self.upsample1(out1)
        out1 = self.conv1(out1)

        out2 = self.downsample2(x)
        out2 = out2 + out1_upsample

        for block in self.convnext_blocks1:
            out2 = block(out2, t)
        out2_upsample = self.upsample2(out2)
        out2 = self.conv2(out2)

        out3 = self.downsample3(x)
        out3 = out3 + out2_upsample

        for block in self.convnext_blocks2:
            out3 = block(out3, t)
        out3 = self.conv3(out3)

        return out3
