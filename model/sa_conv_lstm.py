import torch
import torch.nn as nn

from model.self_attention_memory_modeule import self_attention_memory_module

class SA_ConvLSTM_cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, att_hidden_dim):
        """
                        Initialize ConvLSTM cell.
                        Parameters
                        ----------
                        input_dim: int
                            Number of channels of input tensor.
                        hidden_dim: int
                            Number of channels of hidden state.
                        kernel_size: (int, int)
                            Size of the convolutional kernel.
                        bias: bool
                            Whether to add the bias.
                        att_hidden_dim: int
                            Number of channels of attention hidden state
                        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.attention_layer = self_attention_memory_module(hidden_dim, att_hidden_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=kernel_size,
                      padding=self.padding,
                      bias=self.bias),
            nn.GroupNorm(4 * hidden_dim, 4 * hidden_dim)
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur, m_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        h_next, m_next = self.attention_layer(h_next, m_cur)
        return h_next, (h_next, c_next, m_next)

    # initialize h, c, m
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width)
        c = torch.zeros(batch_size, self.hidden_dim, height, width)
        m = torch.zeros(batch_size, self.hidden_dim, height, width)
        return h, c, m
