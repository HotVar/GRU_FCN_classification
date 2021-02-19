import torch
import torch.nn as nn

class Vanilla_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, dropout_rate, device):
        super(Vanilla_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)

    def forward(self, seq):
        hidden = self.init_hidden()
        # adj_seq = seq.permute(self.batch_size, len(seq), -1)
        output, hidden = self.gru(seq, hidden)
        return output, hidden

class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(channel // reduction, channel, bias=False),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        b, c, s = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1)
        return x * y.expand_as(x)

class FCN_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCN_1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=8,
                               padding=4,
                               padding_mode='replicate'
                               )
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(128, eps=1e-03, momentum=0.99)
        self.SE1 = Squeeze_Excite(128)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels*2,
                               kernel_size=5,
                               padding=2,
                               padding_mode='replicate'
                               )
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps=1e-03, momentum=0.99)
        self.SE2 = Squeeze_Excite(256)

        self.conv3 = nn.Conv1d(in_channels=out_channels*2,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               padding_mode='replicate'
                               )
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-03, momentum=0.99)
        self.gap = nn.AdaptiveAvgPool1d(1)


    def forward(self, seq):
        adj_seq = seq.permute(0, 2, 1)

        y = self.conv1(adj_seq)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.SE1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.SE2(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        y = self.gap(y)

        return y


class GRU_FCN(nn.Module):
    def __init__(self, GRU, FCN, gru_hidden_size, batch_size, seq_len, n_class, dropout_rate):
        super().__init__()
        self.GRU = GRU
        self.FCN = FCN
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_class = n_class

        self.dropout = nn.Dropout(p=dropout_rate)
        self.Dense = nn.Linear(in_features=128, out_features=n_class)

    def forward(self, seq):
        y_GRU, _ = self.GRU(seq.transpose(1, 2)) # dimension shuffle
        # y_GRU, _ = self.GRU(seq)
        y_GRU = y_GRU.transpose(0, 1)[-1]
        y_GRU = self.dropout(y_GRU)
        y_FCN = self.FCN(seq).squeeze()
        if len(y_FCN.size()) == 1:
            y_FCN = y_FCN.unsqueeze(0)
        concat = torch.cat([y_GRU, y_FCN], 1)
        y = self.Dense(concat)
        return y