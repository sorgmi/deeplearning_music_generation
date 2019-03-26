from __future__ import unicode_literals, print_function, division

import random

import numpy as np
import torch
import torch.nn as nn
from music21 import note
from torch import optim
import torch.nn.functional as F
import time
import math

from dataset import entchen
from pytorchseq2seq.AttnDecoderRNN import AttnDecoderRNN
from pytorchseq2seq.DecoderRNN import DecoderRNN
from pytorchseq2seq.EncoderRNN import EncoderRNN
from tools.encodeNotes import getTotalTokens, getStartIndex, getStopIndex, generateInput

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5


def trainSingleExample(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=100):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.shape[0]  # number of timesteps
    target_length = target_tensor.shape[0]

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei,:], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0,0]

    startToken = np.zeros(getTotalTokens())
    startToken[getStartIndex()] = 1
    decoder_input = torch.tensor(startToken, device=device).float()

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di,:].view(1,-1))

            decoder_input = target_tensor[di,:]  # todo: is offset correct?

    else:
        raise NotImplementedError

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train(encoder, decoder, epochs, learning_rate=0.01):

    #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    #decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    piece, notes = entchen.get()

    encoderInput, decoderInput, decoderTarget = generateInput(notes, delta=1)

    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.MultiLabelSoftMarginLoss()

    input_tensor = torch.Tensor(encoderInput)
    target_tensor = torch.Tensor(decoderTarget)

    for i in range(0, epochs):
        loss = trainSingleExample(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print("Epoch", i+1, "finished. Loss:", loss)





##########################
hidden_size = 150
encoder1 = EncoderRNN(getTotalTokens(), hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, getTotalTokens()).to(device)

train(encoder1, decoder1, epochs=100)

torch.save(encoder1, "encoder.pt")
torch.save(decoder1, "decoder.pt")
