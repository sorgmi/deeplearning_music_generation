import music21

majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("G-", 6),("G", 5)])
minors = dict([("F#", 3), ("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("G-", 3),("G", 2)])

def transposePart(part, inPlace=True):
    # part = part.transpose(interval.Interval(part.analyze('key').tonic, pitch.Pitch('C')))
    key = part.analyze('key')
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]
    part.transpose(halfSteps, inPlace=inPlace)

def getNoteList(part, transpose=True):
    notes = []

    if transpose == True:
        #part = part.transpose(interval.Interval(part.analyze('key').tonic, pitch.Pitch('C')))
        key = part.analyze('key')

        if key.mode == "major":
            halfSteps = majors[key.tonic.name]

        elif key.mode == "minor":
            halfSteps = minors[key.tonic.name]

        part.transpose(halfSteps, inPlace=True)

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

def encodeNoteList(notes, delta):
    sequence = []

    for n in notes:
        if (n.isNote):
            sequence.append(n.pitch.midi)
            if n.duration.quarterLength < delta:
                n.duration.quarterLength = delta  # quantization
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
            n.quarterLength = delta
            notes.append(n)
        elif i == 0:
            print("Not implemented Tie start. Index:", index)
            n = music21.note.Note()
            n.pitch.midi = index-128
            n.quarterLength = delta
            notes.append(n)
            #raise NotImplementedError

        else:
            previousNote = notes[-1].pitch.midi

            if index <= 128:
                n = music21.note.Note()
                n.pitch.midi = index
                n.quarterLength = delta
                notes.append(n)
            elif index < 128 * 2 and index - 128 == previousNote:
                notes[-1].quarterLength += delta
            else:
                #print(seq)
                #print(notes)
                #print(index)
                print("Not implemented Tie. Index:", index)
                #raise NotImplementedError


    if input is not None:
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