import music21


def encodeNoteList(notes, delta):
    sequence = []

    for n in notes:
        if (n.isNote):
            sequence.append(n.pitch.midi)
            ticksOn = int(n.duration.quarterLength / delta)
            #print("ticksOn:", ticksOn)
            for i in range(0, ticksOn-1):
                sequence.append(n.pitch.midi + 128)

        if (n.isChord):
            raise NotImplementedError

    return sequence


def split(notes, splitRatio=0.5):
    splitIndex = int(len(notes)*splitRatio)
    x = notes[0:splitIndex]
    y = notes[splitIndex:] + [getStopIndex()]
    return x, y


def decodeSequence(seq, input=None, delta=1):
    notes = []

    for i in range(0, len(seq)):

        index = seq[i]

        if index == getStopIndex():
            break

        if i == 0 and index <= 128:
            n = music21.note.Note()
            n.pitch.midi = index
            notes.append(n)
        elif i == 0:
            print(index)
            raise NotImplementedError

        else:
            previousNote = notes[-1].pitch.midi

            if index <= 128:
                n = music21.note.Note()
                n.pitch.midi = index
                notes.append(n)
            elif index < 128 * 2 and index - 128 == previousNote:
                notes[-1].quarterLength += delta
            else:
                print(seq)
                print(notes)
                print(index)
                raise NotImplementedError


    if input is not None:
        print("reiin", input)
        notes = input + [music21.note.Rest(type='half')] + notes

    piece = music21.stream.Score()
    p1 = music21.stream.Part()
    p1.id = 'part1'

    p1.append(notes)
    piece.insert(0, music21.metadata.Metadata())
    piece.metadata.title = 'Title'
    piece.insert(0, p1)
    return piece, notes


###
def getTotalTokens():
    return 128*2+ 3  # 128 midi notes + Start + Stop + EndOfFrame Tied

def getNoteIndex(n):
    return n.pitch.midi

def getStartIndex():
    return 256

def getStopIndex():
    return 257

def getEOFIndex():
    return 259

def getTiedIndex():
    return 258