import torch
import torch.nn as nn
from .tdln import TDLN
from .attention import Attention
from .simpleLayers import MLP

class ConvNeXt(nn.Module):
    def __init__(self, dim, hidden_dim, input_size):
        super(ConvNeXt, self).__init__()

        self.depthwise_conv = nn.Conv2d(dim,
                                        dim, kernel_size=7,
                                        stride=1,
                                        padding=3,
                                        groups=dim)
        self.tdln = TDLN(input_size)
        self.gelu = nn.GELU()
        self.pw_conv_up = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.pw_conv_down = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        # self.mlp = MLP(dim, [hidden_dim], dim)

    def forward(self, x, t):
        skip = x
        x = self.depthwise_conv(x)
        x = self.tdln(x, t)
        x = self.gelu(x)
        x = self.pw_conv_up(x)
        x = self.gelu(x)
        x = self.pw_conv_down(x)
        # x = self.mlp(x)
        out = x + skip
        return out


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(TransformerBlock, self).__init__()

        self.tdln_1 = TDLN(input_dim)
        self.tdln_2 = TDLN(input_dim)

        self.attn = Attention(input_dim[-1], num_heads)

        flattened_dim = input_dim[0] * input_dim[1]
        self.mlp = MLP(flattened_dim, [2 * flattened_dim], flattened_dim)

    def forward(self, x, t):
        batch_dim, seq_len, embed_dim = x.size()
        skip = x
        x = self.tdln_1(x, t)
        x = self.attn(x)
        x = x + skip
        skip2 = x
        x = self.tdln_2(x, t)
        x = x.view(batch_dim, -1)
        x = self.mlp(x)
        x = x.view(batch_dim, seq_len, embed_dim)
        x = x + skip2
        return x
