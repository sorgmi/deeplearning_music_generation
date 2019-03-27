import random

import music21
import torch
from torch import optim, nn

from dataset import entchen
from pytorchmodels.DecoderRNN import DecoderRNN
from pytorchmodels.EncoderRNN import EncoderRNN
from pytorchmodels.encoding import getStartIndex, getStopIndex, getTotalTokens, encodeNoteList, split, decodeSequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 1.0
MAX_LENGTH = 10


def train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input)
    target_length = len(target)

    input = torch.tensor(input)
    target = torch.tensor(target).view(-1, 1)

    encoder_outputs = torch.zeros(target_length+max_length, encoder.hidden_size, device=device)

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
        raise NotImplementedError
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


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


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


def trainIters(pairs, encoder, decoder, epochs, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter in range(0, epochs):
        for example in range(0, len(pairs)):
            training_pair = pairs[example]
            input = training_pair[0]
            target = training_pair[1]

            loss = train(input, target,encoder,decoder, encoder_optimizer, decoder_optimizer,criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if example % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, example / epochs),
                                             example, example / epochs * 100, print_loss_avg))

            if example % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        print("Epoch", iter+1, " finished. Loss: ", loss)




###########################################################################################################


piece, notes = entchen.get()
input = encodeNoteList(notes, delta=1)
#print([x.pitch.midi for x in notes])
#print(input)
input, target = split(input, splitRatio=0.49)
#print(input, target)
#quit()

pairs = [(input,target)]



hidden_size = 64
encoder = EncoderRNN(getTotalTokens(), hidden_size).to(device)
#attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1,flag=flag).to(device)
decoder = DecoderRNN(hidden_size, getTotalTokens()).to(device)


trainIters(pairs, encoder, decoder, epochs=100, print_every=1000000)


### INFERENCE ###

with torch.no_grad():
    input_tensor = torch.tensor(input)
    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden()

    max_length = 25

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    decoder_input = torch.tensor([[getStartIndex()]], device=device)  # SOS

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        #decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == getStopIndex():
            decoded_words.append(getStopIndex())
            break
        else:
            decoded_words.append(topi.item())

        decoder_input = topi.squeeze().detach()

print("d:",decoded_words)

inputNotes = notes[:int(len(notes)*0.49)]

p, _ = decodeSequence(decoded_words, inputNotes, delta=1)
p.show()

print("i:",input)
print("t:",target)
