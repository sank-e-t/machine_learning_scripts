
import numpy as np
from keras.layers import Conv1D, LSTM, Dense, TimeDistributed, Embedding, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.layers import Bidirectional
from keras.models import load_model

import matplotlib.pyplot as plt


def predict_word(model, word_with_missing, max_word_length, guessed_letters):
    print("LSTM")
    chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = dict((c, i+1) for i, c in enumerate(chars))
    int_to_char = dict((i+1, c) for i, c in enumerate(chars))
    s = ""
    incorrect_guess_vector = [29] * len(chars)
    for w in guessed_letters:
        incorrect_guess_vector[char_to_int[w] - 1] = 30

    for i in guessed_letters:
        s += i
    #guessed_encoded = [char_to_int[char] if char in char_to_int else char_to_int['0'] for char in s]
    word_encoded = [char_to_int[char] if char in char_to_int else char_to_int['0'] for char in word_with_missing]

    word_encoded = np.concatenate((word_encoded, incorrect_guess_vector))
    word_padded = pad_sequences([word_encoded], maxlen=max_word_length+26, padding='post')
    guessed_padded = pad_sequences([word_encoded], maxlen=6, padding='post')
    #word_conc = np.concatenate((word_padded, guessed_padded), axis = 1)
    model2 = load_model("lstm16.h5")
    prediction2 = model2.predict(word_padded)
    predict_vector2 = prediction2[0]

    prediction = model.predict(word_padded)
    predict_vector = prediction[0]

    #print(np.shape(prediction), "prediction", predict_vector2, predict_vector1, predict_vector)
    #print(len(predict_vector), )
    probs = [0]*28

    for i, char in enumerate(word_with_missing):
        if char == '0':

            probabilities = predict_vector[i]
            plt.plot(range(28), probabilities, label = "new")
            plt.plot(range(28), predict_vector2[i], label = "old")
            #print("probs", probabilities)
            probs += probabilities
    plt.legend()
    plt.show()
    predicted_char = ""
    max_prob = 0
    for i in range(2,28):
        if probs[i] > max_prob and int_to_char[i] not in guessed_letters:
            max_prob = probs[i]
            predicted_char = int_to_char[i]

    return predicted_char  # Return None if all characters are guessed or no suitable character is found

word_with_missing = "hang0r"

chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
         'u', 'v', 'w', 'x', 'y', 'z']

guessed_letters = [ 'i']


"""
model = Sequential()
model.add(Embedding(input_dim=len(chars) + 1, output_dim=128, trainable=True))
#model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(len(chars) + 1, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('lstm6_10_seq.h5')
"""
model = load_model("lstm6_10_seq_new.h5")
model.summary()
max_word_length = 35
predicted_char = predict_word(model, word_with_missing, max_word_length, guessed_letters)
print(predicted_char)
"""
import numpy as np
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words

file_path = 'words_250000_train.txt'
words = load_data(file_path)

length = np.zeros(35)
for w in words:
    length[len(w)] += 1
for l in length:

    print(l/250000)

#plt.plot(range(35), length)

#plt.show()
"""
