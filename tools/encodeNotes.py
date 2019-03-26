import music21
from music21 import pitch, interval, stream
import numpy as np

def getNoteList(part, transpose=True):
    notes = []

    if transpose == True:
        part = part.transpose(interval.Interval(part.analyze('key').tonic, pitch.Pitch('C')))

    for x in part.recurse().notes:
        if (x.isNote):
            #print(x.pitch, x.duration.type, x.duration.quarterLength, x.tie, x.octave, x.quarterLength,x.pitch.midi)  # x.seconds not always there
            notes.append(x)

        if (x.isChord):
            print("chord")
            raise NotImplementedError
            for a in x._notes:
                pass
                #print(a.pitch, a.duration.type, a.duration.quarterLength, a.tie, a.octave, a.quarterLength)
            print("endchord")
    return notes

def noteListToInt(notes):
    return [getNoteIndex(x) for x in notes]


def generateInput(notes, split=0.5, delta=0.25, useEOF=False, useTied=False):
    splitIndex = int(len(notes)*split)
    #splitIndex = len(notes) - 1
    input = notes[:splitIndex]
    target = ['start'] + notes[splitIndex:] + ['stop']

    encoderInput = encode(input, delta, useEOF, useTied)
    decoderInput = encode(target, delta, useEOF, useTied)

    # decoder_target_data will be ahead by one timestep and will not include the start character.
    decoderTarget = np.roll(decoderInput, -1, axis=0)
    decoderTarget[-1, :] = 0
    decoderTarget[-1, getStopIndex()] = 1

    return encoderInput, decoderInput, decoderTarget

def encode(notes, delta, useEOF, useTied):
    '''

        :param notes: List of notes (single Part of a piece)
        :param delta: smallest note (quantization)
        :return: 2d array with shape (timesteps, 132)
        '''

    quantization(notes, delta)
    vectorSize = getTotalTokens()

    totalTimesteps = 0
    for x in notes:
        if type(x) == music21.note.Note:
            totalTimesteps += x.quarterLength / delta
        elif x is 'start':
            totalTimesteps += 1
        elif x is 'stop':
            totalTimesteps += 1
        elif type(x) == music21.chord.Chord:
            raise NotImplementedError
        else:
            raise NotImplementedError

    totalTimesteps = int(totalTimesteps)
    if useEOF == True:
        totalTimesteps += len(notes)  # n*EndOFrame symbol
    x = np.zeros( (totalTimesteps, vectorSize) )

    currentTimestep = 0
    for n in notes:
        if n is 'start':
            x[currentTimestep:currentTimestep + 1, getStartIndex()] = 1
            currentTimestep += 1

        elif n is 'stop':
            x[currentTimestep:currentTimestep + 1, getStopIndex()] = 1
            currentTimestep += 1

        elif (n.isNote):
            stepsOn = int(n.quarterLength * (1 / delta))  # todo: rounding issues?
            end = currentTimestep + stepsOn

            x[currentTimestep:end, getNoteIndex(n)] = 1
            if useTied == True:
                x[currentTimestep+1:end, getTiedIndex()] = 1  # Notes are tied
            currentTimestep = end
            if useEOF == True:
                x[currentTimestep, getEOFIndex()] = 1  # EndOfFrame symbol
                currentTimestep += 1

        elif n.isChord:
            raise NotImplementedError  # no chords at the moment
        else:
            raise NotImplementedError

    return x

def quantization(notes, delta):
    for n in notes:
        if type(n) == music21.note.Note and n.quarterLength < delta:
            n.quarterLength = delta
            print("quantization used")


def decodeSequence(seq, input=None):
    # todo: delta & länge beachten
    # todo: use tied
    notes = []
    for i in range(0, len(seq)):
        if seq[i] < 129:
            n = music21.note.Note()
            n.pitch.midi = seq[i]
            notes.append(n)

    if input is not None:
        notes = input + notes

    piece = stream.Score()
    p1 = stream.Part()
    p1.id = 'part1'

    p1.append(notes)
    piece.insert(0, music21.metadata.Metadata())
    piece.metadata.title = 'Title'
    piece.insert(0, p1)
    return piece



def getTotalTokens():
    return 132  # 128 midi notes + Start + Stop + EndOfFrame Tied


def getNoteIndex(n):
    return n.pitch.midi

def getStartIndex():
    return 128

def getStopIndex():
    return 129

def getEOFIndex():
    return 130

def getTiedIndex():
    return 131

