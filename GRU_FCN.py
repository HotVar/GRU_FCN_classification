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
        self.dropout_rate = dropout_rate
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=self.dropout_rate)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)

    def forward(self, seq):
        hidden = self.init_hidden()
        # adj_seq = seq.permute(self.batch_size, len(seq), -1)
        output, hidden = self.gru(seq, hidden)
        return output, hidden


class FCN_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCN_1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=9,
                               padding=4,
                               padding_mode='replicate'
                               )
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels*2,
                               kernel_size=5,
                               padding=2,
                               padding_mode='replicate'
                               )
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=out_channels*2,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               padding_mode='replicate'
                               )
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(128)

        self.gap = nn.AdaptiveAvgPool1d(1)


    def forward(self, seq):
        adj_seq = seq.permute(0, 2, 1)

        y = self.conv1(adj_seq)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        y = self.gap(y)

        return y


class GRU_FCN(nn.Module):
    def __init__(self, GRU, FCN, batch_size, seq_len, n_class, device):
        super().__init__()
        self.GRU = GRU
        self.FCN = FCN
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_class = n_class
        self.device = device

        self.Dense = nn.Linear(in_features=134, out_features=n_class)
        self.softmax = nn.Softmax()

    def forward(self, seq):
        y_GRU, _ = self.GRU(seq)
        y_GRU = y_GRU.squeeze()[-1]
        y_FCN = self.FCN(seq).squeeze()
        concat = torch.cat([y_GRU, y_FCN], 0)
        y = self.Dense(concat)
        y = self.softmax(y)
        return y


