import numpy as np
import torch

from dataset import entchen
from tools.encodeNotes import generateInput, getTotalTokens, getStartIndex, getStopIndex

MAX_LENGTH = 25

piece, notes = entchen.get()
encoderInput, decoderInput, decoderTarget = generateInput(notes, delta=1)

input = torch.Tensor(encoderInput)

encoder = torch.load("encoder.pt")
decoder = torch.load("decoder.pt")

# Encoder Inference
encoder_hidden = encoder.initHidden()
for i in range(input.shape[0]):
    encoder_output, encoder_hidden = encoder(input[i, :], encoder_hidden)

# Decoder Inference
startToken = np.zeros(getTotalTokens())
startToken[getStartIndex()] = 1
decoder_input = torch.tensor(startToken).float()

decoder_hidden = encoder_hidden

decoded_sentence = []
t = -1
while t != getStopIndex() and len(decoded_sentence) < MAX_LENGTH:
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    t = torch.argmax(decoder_output)
    t = t.item()
    decoded_sentence.append(t)
    print(t)

    decoder_input = decoder_output # todo: use one hot encoding?

print(decoded_sentence)