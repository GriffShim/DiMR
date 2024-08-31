import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PosEnc(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len, dim)

    def forward(self, x):
        b, seq_len, dim = x.shape
        pos = torch.arange(seq_len, device=x.device)
        pos = pos.unsqueeze(0)
        pos = self.pos_emb(pos)
        x = x + pos
        return x


class Patchify(nn.Module):
    def __init__(self, patch_size, image_size, embed_dim):
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.image_size = image_size

        num_patches = (image_size // patch_size) ** 2
        self.positional_encoding = PosEnc(num_patches, embed_dim)
    def patchify(self, x):
        return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

    def unpatchify(self, x):
        h = w = self.image_size // self.patch_size
        return rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=self.patch_size, p2=self.patch_size)

    def forward(self, x):
        x = self.patchify(x)
        return self.positional_encoding(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.embed_dim ** -0.5

        self.kqv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def _attention(self, q, k, v):
        attn_mat = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        return attn_mat @ v

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        K, Q, V = self.kqv(x).chunk(3, dim=-1)

        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = self._attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        out = self.out_linear(out)

        return out
