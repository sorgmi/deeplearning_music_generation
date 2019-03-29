from dataset import entchen
from deprecated.encodeNotes import generateInput
from keras.models import Model
from keras.layers import Input, LSTM, Dense

piece, notes = entchen.get()

encoderInput, decoderInput, decoderTarget = generateInput(notes, delta=1)
print(encoderInput.shape, decoderInput.shape, decoderTarget.shape)
encoderInput = encoderInput.reshape( (1,encoderInput.shape[0], encoderInput.shape[1]) )
decoderInput = decoderInput.reshape( (1,decoderInput.shape[0], decoderInput.shape[1]) )
decoderTarget = decoderTarget.reshape( (1,decoderTarget.shape[0], decoderTarget.shape[1]) )
print(encoderInput.shape, decoderInput.shape, decoderTarget.shape)


###
num_encoder_tokens = 132
num_decoder_tokens = num_encoder_tokens
epochs = 100
batch_size = 1
hidden_state_size = 150
###

encoder_input_data = encoderInput
decoder_input_data = decoderInput
decoder_target_data = decoderTarget

### Model ###
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(hidden_state_size, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(hidden_state_size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


#print(model.summary())

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0)





############# INFERENCE #############

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(hidden_state_size,))
decoder_state_input_c = Input(shape=(hidden_state_size,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

def decode_sequence(input_seq):

    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, getStartIndex()] = 1.

    stop_condition = False
    decoded_sentence = []
    tied_sequence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)


        # Tie?
        if output_tokens[0, -1, deprecated.encodeNotes.getTiedIndex()] > 0.5:
         tied_sequence.append(True)
        else:
            tied_sequence.append(False)
        print(output_tokens[0, -1, deprecated.encodeNotes.getTiedIndex()]) #todo: remooove
        output_tokens[0, -1, deprecated.encodeNotes.getTiedIndex()] = 0

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = sampled_token_index
        decoded_sentence.append(sampled_char)

        # todo: set max length
        if (sampled_char == getStopIndex() or len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    print("Ties List:", tied_sequence)

    return decoded_sentence


input_seq = encoder_input_data[0:1]
decoded_sentence = decode_sequence(input_seq)
print('-')
print('Input sentence:', input_seq)
print('Decoded sentence:', decoded_sentence)


x = notes[:int(len(notes)*0.5)]
y = notes[int(len(notes)*0.5):]
from deprecated.encodeNotes import *
p = decodeSequence(decoded_sentence, x + [music21.note.Rest(type='half')])
#p.show()

