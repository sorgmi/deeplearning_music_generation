import glob

from music21 import *
import music21


#####################################
majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("G-", 6),("G", 5)])
minors = dict([("F#", 3), ("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("G-", 3),("G", 2)])

pieces = glob.glob('C:/Users/sorgm/datasets/music21corpus/bach/bwv*.mxl')
for p in pieces:
    piece = converter.parse(p)
    part = piece.parts[0]
    key = part.analyze('key')
    name1 = key.tonic.name
    #print(key, name1)

    if key.mode == "major":
        halfSteps = majors[key.tonic.name]

    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]

    newscore = part.transpose(halfSteps)
    newkey = newscore.analyze('key')
    name2 = newkey.tonic.name
    #print(newkey, name2)

    if newkey.tonic.name != "C" and newkey.tonic.name != "A":
        print(p)
        print(key, name1, "-->", newkey, name2)
        #quit()