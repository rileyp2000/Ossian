import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.module):
    def __init__(self, char_len, n_hidden=256, n_layers=2):
        super.__init__()
        self.hidden = n_hidden
        self.layers = n_layers
        self.drop = .4
        self.lr = le-4

        self.int2char = dict(enumerate(chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.lstm = nn.LSTM(char_len, self.hidden, self.layers, dropout=self.drop, batch_first=True)
        self.dropout = nn.Dropout(self.drop)
        self.linear = nn.Linear(n_hidden, char_len)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.linear(out)
        out = out.contiguous().view(-1, self.hidden)

        return out, hidden

def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.layers, batch_size, self.hidden).zero_().cuda(),
                  weight.new(self.layers, batch_size, self.hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.layers, batch_size, self.hidden).zero_(),
                      weight.new(self.layers, batch_size, self.hidden).zero_())

        return hidden
