import numpy as np
import torch

from dataset import entchen
from tools.encodeNotes import generateInput, getTotalTokens, getStartIndex, getStopIndex

MAX_LENGTH = 35

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
decoder_input = torch.Tensor(startToken).float()

decoder_hidden = encoder_hidden

decoded_sentence = []
t = -1
while t != getStopIndex() and len(decoded_sentence) < MAX_LENGTH:
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    t = torch.argmax(decoder_output)
    t = t.item()
    decoded_sentence.append(t)
    #print(t)

    #decoder_input = decoder_output

    decoder_input = torch.zeros((1,getTotalTokens()) )
    decoder_input[0,t] = 1

print(decoded_sentence, len(decoded_sentence))


x = notes[:int(len(notes)*0.5)]
y = notes[int(len(notes)*0.5):]
from tools.encodeNotes import *
p = decodeSequence(decoded_sentence, x + [music21.note.Rest(type='half')])
p.show()

x = [x.pitch.midi for x in x]
y = [y.pitch.midi for y in y]
#print(x)
print(y)