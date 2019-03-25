import torch
import torch.nn.functional as F
from torch import nn

from tools.encodeNotes import getTotalTokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanillaLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(VanillaLSTM, self).__init__()
        self.input_size = input_size

        self.hidden_size = input_size

        self.lstm = nn.LSTM(input_size, self.hidden_size)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size)
        #self.lstm3 = nn.LSTM(self.hidden_size, self.hidden_size)

        #self.out0 = nn.Linear(self.hidden_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        #self.softmax = nn.Softmax()
        #self.m = nn.Sigmoid()


    def forward(self, input, hidden):
        output = input

        #output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output, hidden = self.lstm2(output, hidden)
        #output, hidden = self.lstm3(output, hidden)
        #output = self.out0(output[-1])

        output = self.out(output[-1])

        #output = self.softmax(self.out(output[0]))
        #output = self.out(output[0])
        #output = self.m(output)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))