import torch
import torch.nn as nn


class self_attention_memory_module(nn.Module):
    # SAM 自注意力模块
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.layer_q = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, 1)

        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)

        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)

    def forward(self, h, m):
        batch_size, channels, H, W = h.shape
        # **********************  feature aggregation ******************** #

        # Use 1x1 convolution for Q,K,V Generation
        K_h = self.layer_k(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)

        Q_h = self.layer_q(h)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)

        K_m = self.layer_k2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)

        V_m = self.layer_v2(m)
        V_m = V_m.View(batch_size, self.input_dim, H * W)

        # **********************  hidden h attention ******************** #
        # [batch_size,H*W,H*W]
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)

        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        # **********************  memory m attention ******************** #
        # [batch_size,H*W,H*W]
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)

        Z_m = torch.matmul(A_m, V_m.permate(0, 2, 1))
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)  # [batch_size,in_channels*2,H,W]

        # Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim=1))
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        #
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m
