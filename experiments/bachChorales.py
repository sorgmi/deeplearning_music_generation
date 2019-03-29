import glob
import math
import random
import time

import music21
from music21 import *
import torch
from torch import optim, nn

from dataset import entchen
from pytorchmodels.AttnDecoderRNN import AttnDecoderRNN
from pytorchmodels.DecoderRNN import DecoderRNN
from pytorchmodels.EncoderRNN import EncoderRNN
from pytorchmodels.encoding import getStartIndex, getStopIndex, getTotalTokens, encodeNoteList, decodeSequence, \
    transposePart, getNoteList
from pytorchmodels.inference import evaluate
from pytorchmodels.training import train, trainIters


def getPartByName(piece, filter="soprano"):
    for p in piece.parts:  # type: music21.stream.Part
        if p.partName.lower() == filter.lower():
            return p
    return None

def parseMXLfiles(pathPattern, filter="soprano"):
    files = glob.glob(pathPattern)
    notes = []
    for f in files:
        piece = converter.parse(f)
        p = getPartByName(piece, filter)

        #p.show()
        #print(p.analyze('key'))
        transposePart(p, inPlace=True)
        #print(p.analyze('key'))
        #p.show()
        #quit()

        if p is None:
            print("no part found in", f, "with filter", filter)

        notes.append(getNoteList(p, transpose=False))

    return notes

def generateTrainingData(notes, delta, split):
    max = 0
    data = []
    for n in notes:
        splitIndex = int(len(n)*split)
        x = n[0:splitIndex]
        y = n[splitIndex:]
        input = encodeNoteList(x,delta)
        data.append( (input, encodeNoteList(y,delta)+ [getStopIndex()], x, y) )
        if len(input) > max:
            max = len(input)
    return data, max

### Data preparation ###
delta = 0.5
splitFactor = 0.5
notes = parseMXLfiles('C:/Users/sorgm/datasets/music21corpus/bach/bwv3.6.mxl')

encodedNotes, MAX_LENGTH = generateTrainingData(notes, delta, splitFactor)
#print(encodedNotes)


### Training ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 128
encoder = EncoderRNN(getTotalTokens(), hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, getTotalTokens(), max_length=MAX_LENGTH, dropout_p=0.1).to(device)
#decoder = DecoderRNN(hidden_size, getTotalTokens()).to(device)

trainIters(encodedNotes, encoder, decoder, epochs=200, print_every=2, plot_every=2, max_length=MAX_LENGTH)

### Inference ###
sampleIndex = 0
pair = encodedNotes[sampleIndex]
input = pair[0]
target = pair[1]

decoded_words, decoder_attentions = evaluate(input, encoder, decoder, MAX_LENGTH)

inputNotes = encodedNotes[sampleIndex]
inputNotes = inputNotes[2]

p, _ = decodeSequence(decoded_words, inputNotes, delta=delta)
p.show()

print("i:", input)
print("d:", decoded_words)
print("t:", target)
