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
from pytorchVanillaRNN.VanillaLSTM import VanillaLSTM
from tools.encodeNotes import getTotalTokens, getStartIndex, getStopIndex, generateInput
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################
hidden_size = 256
model = VanillaLSTM(getTotalTokens(), getTotalTokens()).to(device)

piece, notes = entchen.get()
encoderInput, decoderInput, decoderTarget = generateInput(notes, delta=1)

input = torch.tensor(encoderInput).float()
target = torch.tensor(decoderTarget).float()
print(input.shape, target.shape)

input = input.view(16,1,132)  # seq.length,batch_size,dimensions
print(input.shape, target.shape)

output, hidden = model(input, model.initHidden())
print(output.shape)
#print(hidden)

optimizer = optim.SGD(model.parameters(), lr=0.01)
#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()


epochs = 50
print("\nTraining start...")
for e in range(0, epochs):
    optimizer.zero_grad()
    #model.zero_grad()

    output, hidden = model(input, model.initHidden())

    loss = 0

    for i in range(0, target.shape[0]):
        output, hidden = model(output.view(1,1,-1), hidden)
        #print(output.shape)

        t = target[i]
        t = torch.argmax(t)
        t = torch.Tensor([t])
        #print("loo:", output.shape, t.shape)
        loss += criterion(output, t.long())

        # loss += criterion(decoder_output, target[i])

    loss.backward(retain_graph=True)
    optimizer.step()
    print("epoch", e+1, ". Loss:", loss.item())




### INFERENCE ###
print("\nInference:")
MAX_LENGTH = 35

output, hidden = model(input, model.initHidden())

decoded_sentence = []
t = -1
while t != getStopIndex() and len(decoded_sentence) < MAX_LENGTH:
    output, hidden = model(output.view(1,1,-1), hidden)
    t = torch.argmax(output)
    t = t.item()
    decoded_sentence.append(t)
    #print(t)


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








