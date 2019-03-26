import torch
import torch.nn.functional as F
from torch import nn

from tools.encodeNotes import getTotalTokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(getTotalTokens(), hidden_size)
        #self.gru2 = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax(dim=1)
        #self.softmax = nn.Softmax()
        #self.m = nn.Sigmoid()


    def forward(self, input, hidden):
        output = input.view(1, 1, -1)

        #output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #output, hidden = self.gru2(output, hidden)

        #output = self.softmax(self.out(output[0]))
        output = self.out(output[0])
        #output = self.m(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)