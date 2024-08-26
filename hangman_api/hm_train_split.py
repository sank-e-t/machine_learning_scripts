
import numpy as np
import matplotlib.pyplot as plt
"""training model"""

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
import keras.saving
import random
# a_i_m_l
# animal

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words


def simulate_missing_letters(word, char_to_int, p_missing=0.4):
    """
    Randomly replace letters with '0' to simulate missing letters,
    with p_missing chance for each letter.
    """

    rand_dic = {}

    for i in word:
        rand_dic[i] = rand_dic.get(i, random.random())
    simulated_words = [''] * 4
    for i in range(4):
        for char in word:
            simulated_words[i] += '0' if rand_dic[char] < p_missing - 0.2 * i else char

    return simulated_words


def preprocess_data(words, max_word_length, p_missing=0.4):
    sequences = []
    targets = []
    chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = dict((c, i + 1) for i, c in enumerate(chars))
    int_to_char = dict((i + 1, c) for i, c in enumerate(chars))
    # feature  0.6-> 0.4 0.4->0.2 0.2 ->0.0

    for i in range(1):
        for word in words:
            simulated_words = simulate_missing_letters(word, char_to_int, 0.6)
            sequences.append(
                [char_to_int[char] if char in char_to_int else char_to_int['0'] for char in simulated_words[0]])
            targets.append([char_to_int[char] for char in word])
            sequences.append(
                [char_to_int[char] if char in char_to_int else char_to_int['0'] for char in simulated_words[1]])
            targets.append([char_to_int[char] for char in word])
            sequences.append(
                [char_to_int[char] if char in char_to_int else char_to_int['0'] for char in simulated_words[2]])
            targets.append([char_to_int[char] for char in word])

    sequences = pad_sequences(sequences, maxlen=max_word_length, padding='post')
    targets = pad_sequences(targets, maxlen=max_word_length, padding='post')
    return sequences, targets


def train(words):
    # words = api.full_dictionary
    max_word_length = 35  # max(len(word) for word in words)  # finding largest word, maybe switch it to 50 or something
    random.shuffle(words)  # removing the alphabetic order so that the model trains impartially

    words1_5 = []
    words6_10 = []
    words11_15 = []
    words16 = []
    for word in words:
        if len(word) < 6:
            words1_5.append(word)
        if len(word) < 11:
            words6_10.append(word)
        if len(word) < 16 and len(word) >5:
            words11_15.append(word)
        if len(word) > 11:
            words16.append(word)


    chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = dict((c, i + 1) for i, c in enumerate(chars))
    int_to_char = dict((i + 1, c) for i, c in enumerate(chars))
    print(char_to_int)
    """
    # -----------------------------------------------------------------------------------------
    X, y = preprocess_data(words1_5, max_word_length, p_missing=0.4)
    y = to_categorical(y, num_classes=None)
    bs = 64
    epoch = 15

    model = Sequential()
    model.add(Embedding(input_dim=len(chars)+1, output_dim=128, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(len(chars)+1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.25, epochs= epoch, batch_size=bs ,
              callbacks=[early_stopping])
    model.save("lstm1_5_seq.h5")
    #keras.saving.save_model("/lstm1_5.keras")
    # -----------------------------------------------------------------------------------------
    """
    X, y = preprocess_data(words6_10, max_word_length, p_missing=0.4)
    y = to_categorical(y, num_classes=None)
    bs = 32
    epoch = 15

    model = Sequential()
    model.add(Embedding(input_dim=len(chars) + 1, output_dim=128, trainable=True))
    # model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(len(chars) + 1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.25, epochs=epoch, batch_size=bs,
              callbacks=[early_stopping])
    model.save("lstm6_10_seq.h5")
    # keras.saving.save_model("/lstm1_5.keras")

    """
    # -----------------------------------------------------------------------------------------

    X, y = preprocess_data(words11_15, max_word_length, p_missing=0.4)
    y = to_categorical(y, num_classes=None)
    bs = 64
    epoch = 15

    model = Sequential()
    model.add(Embedding(input_dim=len(chars) + 1, output_dim=128, trainable=True))
    #model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(len(chars) + 1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)
    model.fit(X, np.array(y), validation_split=0.25, epochs=epoch, batch_size=bs,
              callbacks=[early_stopping])
    model.save("lstm11_15_seq.h5")
    # keras.saving.save_model("/lstm1_5.keras")
    # -----------------------------------------------------------------------------------------

    X, y = preprocess_data(words16, max_word_length, p_missing=0.4)
    y = to_categorical(y, num_classes=None)
    bs = 64
    epoch = 15
    model = Sequential()
    model.add(Embedding(input_dim=len(chars) + 1, output_dim=128, trainable=True))
    #model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(len(chars) + 1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.25, epochs=epoch, batch_size=bs,
              callbacks=[early_stopping])
    model.save("lstm16_seq.h5")
    # keras.saving.save_model("/lstm1_5.keras")

    # -----------------------------------------------------------------------------------------
    """



file_path = 'words_250000_train.txt'
words = load_data(file_path)
train(words)


def predict_word(model, word_with_missing, max_word_length, guessed_letters):
    print("LSTM")
    chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    word_encoded = [char_to_int[char] if char in char_to_int else char_to_int['0'] for char in word_with_missing]

    word_padded = pad_sequences([word_encoded], maxlen=max_word_length, padding='post')

    prediction = model.predict(word_padded)
    predict_vector = prediction[0]
    print(len(prediction), "prediction")
    print(len(predict_vector), )

    for i, char in enumerate(word_with_missing):
        if char == '0':
            probabilities = predict_vector[i]
            print("probs", probabilities)
            sorted_indices = np.argsort(-probabilities)

            for idx in sorted_indices:
                predicted_char = int_to_char[idx]
                if predicted_char != '0' and predicted_char not in guessed_letters:
                    return predicted_char
    return None  # Return None if all characters are guessed or no suitable character is found


"""
word_with_missing = "an000"

guessed_letters = set(['e', 'o', 'i', 's', 't', 'd', 'a', "g"])
model = load_model('lstm1_5.h5')
max_word_length = 5
predicted_char = predict_word(model, word_with_missing, max_word_length, guessed_letters)
print(predicted_char)

"""