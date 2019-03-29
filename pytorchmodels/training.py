import math
import random
import time

import torch
from torch import optim, nn

from pytorchmodels.encoding import getStartIndex, getStopIndex, getTotalTokens, encodeNoteList, split, decodeSequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 1.0

def train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input)
    target_length = len(target)

    input = torch.tensor(input)
    target = torch.tensor(target).view(-1, 1)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[getStartIndex()]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target[di])
            decoder_input = target[di]  # Teacher forcing

    else:
        #raise NotImplementedError
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target[di])
            if decoder_input.item() == getStopIndex():
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(pairs, encoder, decoder, epochs, max_length, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    numberofExamples = len(pairs)
    for iter in range(0, epochs):
        for example in range(0, numberofExamples):
            training_pair = pairs[example]
            input = training_pair[0]
            target = training_pair[1]

            loss = train(input, target,encoder,decoder, encoder_optimizer, decoder_optimizer,criterion, max_length=max_length)
            print_loss_total += loss
            plot_loss_total += loss

            if example % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, example / numberofExamples),
                                             example, example / numberofExamples * 100, print_loss_avg))

            if example % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        print("Epoch", iter+1, " finished. Loss:", loss)

    #showPlot(plot_losses)


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


import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


