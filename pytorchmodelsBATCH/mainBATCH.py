import glob
import time

import music21
from music21 import *
import torch

from pytorchmodelsBATCH.AttnDecoderRNN import AttnDecoderRNN
from pytorchmodelsBATCH.DecoderRNN import DecoderRNN
from pytorchmodelsBATCH.EncoderRNN import EncoderRNN
from pytorchmodelsBATCH.encoding import getStartIndex, getStopIndex, getTotalTokens, encodeNoteList, decodeSequence, \
    transposePart, getNoteList, getPadIndex
from pytorchmodelsBATCH.inference import evaluate
from pytorchmodelsBATCH.training import train, trainIters


def getPartByName(piece, filter="soprano"):
    for p in piece.parts:  # type: music21.stream.Part
        if p.partName.lower() == filter.lower():
            return p
    return None

def parseMXLfiles(pathPattern, filter="soprano"):
    files = glob.glob(pathPattern)
    print(files)
    notes = []
    for f in files:
        piece = converter.parse(f)
        p = getPartByName(piece, filter)

        if p is None:
            print("no part found in", f, "with filter", filter)
            continue

        #p.show()
        #print(p.analyze('key'))
        transposePart(p, inPlace=True)
        #print(p.analyze('key'))
        #p.show()
        #quit()

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


def padBatch(pairs):
    maxIn = 0
    maxTarget = 0
    input = []
    target = []
    for p in pairs:
        if len(p[0]) > maxIn:
            maxIn = len(p[0])
        if len(p[1]) > maxTarget:
            maxTarget = len(p[1])


    for p in pairs:
        padsIn =  maxIn - len(p[0])
        padsTarget = maxTarget - len(p[1])

        input.append( p[0] + [getPadIndex()]*padsIn)
        target.append( p[1] + [getStopIndex()] * padsTarget)
        #print(input[-1], target[-1])

    input = torch.tensor(input)
    target = torch.tensor(target)

    return input, target


### Data preparation ###
delta = 0.5
splitFactor = 0.5
#notes = parseMXLfiles('C:/Users/sorgm/datasets/music21corpus/bach/bwv3.6.mxl')
notes = parseMXLfiles('C:/Users/sorgm/datasets/music21corpus/bach/bwv1*.mxl')
#print(len(notes))
#quit()

encodedNotes, MAX_LENGTH = generateTrainingData(notes, delta, splitFactor)
#print(encodedNotes)

print("loaded", len(encodedNotes), "training points")
train = encodedNotes[0:50]
test = encodedNotes[50:100]
encodedNotes = train
#quit()

input, target = padBatch(encodedNotes)
print("batch shapes:", input.shape, target.shape)
#quit()


### Training ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 128
encoder = EncoderRNN(getTotalTokens(), hidden_size).to(device)
#decoder = AttnDecoderRNN(hidden_size, getTotalTokens(), max_length=MAX_LENGTH, dropout_p=0.1).to(device)
decoder = DecoderRNN(hidden_size, getTotalTokens()).to(device)


batches = [(input, target)]

s = time.time()
trainIters(batches, encoder, decoder, epochs=10, print_every=2, plot_every=2, max_length=MAX_LENGTH)
print(time.time()-s, len(train))


quit()


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
