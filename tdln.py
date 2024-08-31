import torch
import torch.nn as nn
import torch.nn.functional as F


class TDLN(nn.Module):
    def __init__(self, feature_dim):
        super(TDLN, self).__init__()
        self.feature_dim = feature_dim

        self.w = nn.Parameter(torch.empty(1))
        self.b = nn.Parameter(torch.empty(1))
        nn.init.normal_(self.w, mean=0.0, std=0.02)
        nn.init.zeros_(self.b)

        self.p1 = nn.Parameter(torch.randn(1, *feature_dim))
        self.p2 = nn.Parameter(torch.randn(1, *feature_dim))
        self.p3 = nn.Parameter(torch.randn(1, *feature_dim))
        self.p4 = nn.Parameter(torch.randn(1, *feature_dim))

    def forward(self, x, t):
        s_t = torch.sigmoid(self.w * t + self.b)
        for _ in range(len(self.feature_dim)):
            s_t = s_t.unsqueeze(-1)
        gamma = s_t * self.p1 + (1 - s_t) * self.p2
        beta = s_t * self.p3 + (1 - s_t) * self.p4

        normalized_x = F.layer_norm(x, x.shape[1:])
        out = gamma * normalized_x + beta

        return out
