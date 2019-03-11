import music21
from music21 import stream, note, metadata

def get():
    piece = stream.Score()
    p1 = stream.Part()
    p1.id = 'part1'

    notes = [note.Note('C4', type='quarter'),
             note.Note('D4', type='quarter'),
             note.Note('E4', type='quarter'),
             note.Note('F4', type='quarter'),
             note.Note('G4', type='half'),
             note.Note('G4', type='half'),

             note.Note('A4', type='quarter'),
             note.Note('A4', type='quarter'),
             note.Note('A4', type='quarter'),
             note.Note('A4', type='quarter'),
             note.Note('G4', type='half'),

             note.Note('A4', type='quarter'),
             note.Note('A4', type='quarter'),
             note.Note('A4', type='quarter'),
             note.Note('A4', type='quarter'),
             note.Note('G4', type='half'),

             note.Note('F4', type='quarter'),
             note.Note('F4', type='quarter'),
             note.Note('F4', type='quarter'),
             note.Note('F4', type='quarter'),
             note.Note('E4', type='half'),
             note.Note('E4', type='half'),

             note.Note('D4', type='quarter'),
             note.Note('D4', type='quarter'),
             note.Note('D4', type='quarter'),
             note.Note('D4', type='quarter'),
             note.Note('C4', type='half'),
             ]

    p1.append(notes)

    piece.insert(0, metadata.Metadata())
    piece.metadata.title = 'Alle meine Entchen'
    piece.insert(0, p1)
    return piece, notes


#get().show('midi')
#get().show()


