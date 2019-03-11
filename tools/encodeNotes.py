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
    return [x.pitch.midi for x in notes]


def generateInput(notes, split=0.5, delta=0.25):
    splitIndex = int(len(notes)*split)
    #splitIndex = len(notes) - 1
    input = notes[:splitIndex]
    target = ['start'] + notes[splitIndex:] + ['stop']

    encoderInput = encode(input, delta)
    decoderInput = encode(target, delta)

    # decoder_target_data will be ahead by one timestep and will not include the start character.
    decoderTarget = np.roll(decoderInput, -1, axis=0)
    decoderTarget[-1, :] = 0
    decoderTarget[-1, getStopIndex()] = 1

    return encoderInput, decoderInput, decoderTarget

def encode(notes, delta):
    '''

        :param notes: List of notes (single Part of a piece)
        :param delta: smallest note (quantization)
        :return: 2d array with shape (131, timesteps)
        '''
    # todo: gebundene noten i, input

    quantization(notes, delta)

    vectorSize = getTotalTokens()

    # todo: measure timesteps in delta units?
    totalTimesteps = 0
    for x in notes:
        if type(x) == music21.note.Note:
            totalTimesteps += x.quarterLength / delta
        elif x is 'start':
            totalTimesteps += 1
        elif x is 'stop':
            totalTimesteps += 1
    totalTimesteps = int(totalTimesteps) #todo: aufrunden?

    # Todo: use EOF symbol
    #totalTimesteps += len(notes)  # Start & End Symbol + n*EndOFrame symbol
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
            currentTimestep = end
            # Todo: activate EOF symbol
            # x[getEOFIndex(), currentTimestep] = 1 # EndOfFrame symbol

        elif n.isChord:
            raise NotImplementedError  # no chords at the moment
        else:
            raise NotImplementedError

    return x

def quantization(notes, delta):
    for n in notes:
        if n.quarterLength < delta:
            n.quarterLength = delta
            print("quantization used")


def decodeSequence(seq, input=None):
    #todo: delta & länge beachten
    #todo: gebunden
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
    return 131  # 128 midi notes + Start + Stop + EndOfFrame


def getNoteIndex(n):
    # todo: tied?
    return n.pitch.midi

def getStartIndex():
    return 128

def getStopIndex():
    return 129

def getEOFIndex():
    return 130

