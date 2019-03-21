from __future__ import unicode_literals, print_function, division

import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import math

from dataset import entchen
from pytorchmodels.AttnDecoderRNN import AttnDecoderRNN
from pytorchmodels.DecoderRNN import DecoderRNN
from pytorchmodels.EncoderRNN import EncoderRNN
from tools.encodeNotes import getTotalTokens, getStartIndex, getStopIndex, generateInput

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=100):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.shape[1]
    target_length = target_tensor.shape[1]

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[0,ei,:], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    startToken = np.zeros(getTotalTokens())
    startToken[getStartIndex()] = 1
    decoder_input = torch.tensor(startToken, device=device).float()

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = True

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            #print("crit:",decoder_output.shape, target_tensor[0,di,:].shape)
            #TODO: Use multiclass loss (Tied) ???
            #todo: https://pytorch.org/docs/stable/nn.html#bceloss (or #bcewithlogitsloss)
            t = target_tensor[0,di,:]
            t = torch.argmax(t, dim=None)
            #print(t.shape,t)
            t = torch.Tensor([t])
            loss += criterion(decoder_output, t.long())
            decoder_input = target_tensor[0,di,:]  # Teacher forcing

    else:
        raise NotImplementedError

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent != 0:
        es = s / (percent)
    else:
        es = 0
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    piece, notes = entchen.get()

    encoderInput, decoderInput, decoderTarget = generateInput(notes, delta=1)
    print(encoderInput.shape, decoderInput.shape, decoderTarget.shape)
    encoderInput = encoderInput.reshape((1, encoderInput.shape[0], encoderInput.shape[1]))
    decoderInput = decoderInput.reshape((1, decoderInput.shape[0], decoderInput.shape[1]))
    decoderTarget = decoderTarget.reshape((1, decoderTarget.shape[0], decoderTarget.shape[1]))
    print(encoderInput.shape, decoderInput.shape, decoderTarget.shape)


    criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    #todo: set range
    for iter in range(0, 1):
        input_tensor = torch.Tensor(encoderInput)
        target_tensor = torch.Tensor(decoderTarget)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)

##########################
hidden_size = 256
encoder1 = EncoderRNN(getTotalTokens(), hidden_size).to(device)
attn_decoder1 = DecoderRNN(hidden_size, getTotalTokens()).to(device)

trainIters(encoder1, attn_decoder1, 5, print_every=1)
