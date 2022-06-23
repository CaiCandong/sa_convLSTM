import torch
import torch.nn as nn
import random
from model.sa_convlstm.SAConvLSTM import SAConvLSTMCell


class Encode2Decode(nn.Module):
    """自回归,前t时刻预测后t时刻;无法做到语言翻译的效果,输入输出shape相同"""

    # self-attention convlstm for spatiotemporal prediction model
    def __init__(self, input_dim, hidden_dim, attn_hidden_dim, kernel_size, img_size=(16, 16), num_layers=4,
                 batch_first=True,
                 bias=True,
                 ):
        super(Encode2Decode, self).__init__()
        self.img_size = img_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.attn_hidden_dim = attn_hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        # encode:降低图片分辨率
        self.img_encode = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        # encode:还原图片分辨率
        self.img_decode = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, (3, 3), (2, 2), (1, 1), output_padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, (3, 3), (2, 2), (1, 1), output_padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, input_dim, (1, 1), (1, 1)),
        )
        cell_list, bns = [], []
        for i in range(0, self.num_layers):
            # cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cur_input_dim = self.hidden_dim
            cell_list.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            att_hidden_dim=self.attn_hidden_dim,
                                            bias=self.bias,
                                            ))
            # Use layer norm
            bns.append(nn.LayerNorm(normalized_shape=[hidden_dim, *self.img_size]))

        self.cell_list = nn.ModuleList(cell_list)
        self.bns = nn.ModuleList(bns)

        # Linear
        self.decoder_predict = nn.Conv2d(in_channels=hidden_dim,
                                         out_channels=input_dim,
                                         kernel_size=(1, 1),
                                         padding=(0, 0))

    def forward(self, x, y, teacher_forcing_rate=0.5, hidden_state=None):
        if not self.batch_first:
            # (t,b,c,h,w)->(b,t,c,h,w)
            x = x.permute(1, 0, 2, 3, 4)
            y = y.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = x.shape
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h // 4, w // 4),device=x.device)
        seq_len, horizon = x.size(1), y.size(1)
        predict_temp_de = []
        frames = torch.cat([x, y], dim=1)
        for t in range(seq_len + horizon - 1):
            if t < seq_len or random.random() < teacher_forcing_rate:
                x = frames[:, t, :, :, :]
            else:
                x = out
            x = self.img_encode(x)
            for i, cell in enumerate(self.cell_list):
                h_next, c_next, m_next = cell(x, hidden_state[i])
                out, hidden_state[i] = h_next, (h_next, c_next, m_next)
                out = self.bns[i](out)
            out = self.img_decode(out)
            predict_temp_de.append(out)
        predict_temp_de = torch.stack(predict_temp_de, dim=1)
        predict_temp_de = predict_temp_de[:, seq_len - 1:, :, :, :]
        return predict_temp_de

    def _init_hidden(self, batch_size, image_size,device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size,device))
        return init_states


def main():
    model = Encode2Decode(1, 16, 16, (3, 3))
    img_size = 1, 64, 64
    batch_size, seq_len, horizon = 2, 10, 10
    x = torch.rand(batch_size, seq_len, *img_size)
    y = torch.rand(batch_size, horizon, *img_size)
    y_hat = model(x, y)
    print(y_hat.shape)


if __name__ == '__main__':
    main()
