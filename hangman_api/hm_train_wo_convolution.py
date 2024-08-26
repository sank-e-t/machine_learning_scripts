
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


def simulate_incorrect_guesses(word):
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    # Character probabilities based on frequency
    char_prob = [0.0846392933555352, 0.018750476527547293, 0.042060086240896556, 0.03523056402977109,
                 0.11001079658462706, 0.012439604545672753, 0.02440291686629837, 0.027321383355939956,
                 0.08694968716260673, 0.00178374262147099, 0.008793992317199326, 0.05762147569638912,
                 0.0292698515493146, 0.07165985957850962, 0.0706211471865343, 0.030961347850519544,
                 0.0018759889417370359, 0.07023333612582398, 0.06987282244560056, 0.06460866381205095,
                 0.03638270174411436, 0.009910361050214943, 0.008345468117130236, 0.0028473991714774377,
                 0.019289364469917816, 0.004117668653100182]

    incorrect_guess_vector = [30] * len(chars)
    for c in word:
        char_prob[char_to_int[c]] = 0

    for _ in range(random.randint(1, 5)):  # Randomly choose 1-5 incorrect guesses
        incorrect_char = random.choices(chars, weights=char_prob, k=1)[0]  # Select based on weighted probability

        if incorrect_char not in word:
            incorrect_guess_vector[char_to_int[incorrect_char]] = 31  # Mark as incorrect guess

    return incorrect_guess_vector

def preprocess_data(words, max_word_length, p_missing=0.4):
    sequences = []
    targets = []
    incorrect_guesses = []
    chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = dict((c, i + 1) for i, c in enumerate(chars))
    int_to_char = dict((i + 1, c) for i, c in enumerate(chars))
    # feature  0.6-> 0.4 0.4->0.2 0.2 ->0.0

    for i in range(1):
        for word in words:
            simulated_words = simulate_missing_letters(word, char_to_int, 0.6)
            sequences.append([char_to_int[char] if char in char_to_int else char_to_int['0'] for char in simulated_words[0]])
            targets.append([char_to_int[char] for char in word])
            incorrect_guesses.append(simulate_incorrect_guesses(word))

            sequences.append([char_to_int[char] if char in char_to_int else char_to_int['0'] for char in simulated_words[1]])
            targets.append([char_to_int[char] for char in word])
            incorrect_guesses.append(simulate_incorrect_guesses(word))

            sequences.append([char_to_int[char] if char in char_to_int else char_to_int['0'] for char in simulated_words[2]])
            targets.append([char_to_int[char] for char in word])
            incorrect_guesses.append(simulate_incorrect_guesses(word))

    sequences = pad_sequences(sequences, maxlen=max_word_length, padding='post', value = -1)
    targets = pad_sequences(targets, maxlen=max_word_length, padding='post',value = -1)
    return np.array(sequences), targets, np.array(incorrect_guesses)


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
        elif len(word) < 16:
            words11_15.append(word)
        else:
            words16.append(word)

    chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = dict((c, i + 1) for i, c in enumerate(chars))
    int_to_char = dict((i + 1, c) for i, c in enumerate(chars))
    print(char_to_int)

    # -----------------------------------------------------------------------------------------
    X, y, incorrect_guesses = preprocess_data(words1_5, max_word_length, p_missing=0.4)
    y = to_categorical(y, num_classes=None)
    X = np.concatenate((X, incorrect_guesses), axis=1)
    bs = 64
    epoch = 15
    embedding_dim = 128

    model = Sequential()
    model.add(Embedding(input_dim=len(chars)+3, output_dim=128, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(chars)+1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.2, epochs= epoch, batch_size=bs ,
              callbacks=[early_stopping])
    model.save("lstm1_5_guesses.h5")
    #keras.saving.save_model("/lstm1_5.keras")
    # -----------------------------------------------------------------------------------------
    X, y, incorrect_guesses = preprocess_data(words6_10, max_word_length, p_missing=0.4)
    y = to_categorical(y, num_classes=None)
    X = np.concatenate((X, incorrect_guesses), axis=1)
    bs = 64
    epoch = 15

    model = Sequential()
    model.add(Embedding(input_dim=len(chars)+3, output_dim=128, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(chars)+1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.2, epochs= epoch, batch_size=bs ,
              callbacks=[early_stopping])
    model.save("lstm6_10_seq_guesses.h5")
    #keras.saving.save_model("/lstm1_5.keras")

    # -----------------------------------------------------------------------------------------

    X, y, incorrect_guesses = preprocess_data(words11_15, max_word_length, p_missing=0.4)
    y = to_categorical(y, num_classes=None)
    X = np.concatenate((X, incorrect_guesses), axis=1)
    bs = 64
    epoch = 15

    model = Sequential()
    model.add(Embedding(input_dim=len(chars) + 1, output_dim=128, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(chars) + 1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.2, epochs=epoch, batch_size=bs,
              callbacks=[early_stopping])
    model.save("lstm11_15_seq_guesses.h5")
    # keras.saving.save_model("/lstm1_5.keras")
    # -----------------------------------------------------------------------------------------

    X, y, incorrect_guesses = preprocess_data(words16, max_word_length, p_missing=0.4)
    y = to_categorical(y, num_classes=None)
    X = np.concatenate((X, incorrect_guesses), axis=1)
    bs = 64
    epoch = 15

    model = Sequential()
    model.add(Embedding(input_dim=len(chars) + 1, output_dim=128, trainable=True))
    #model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(chars) + 1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.2, epochs=epoch, batch_size=bs,
              callbacks=[early_stopping])
    model.save("lstm16_seq_guesses.h5")
    # keras.saving.save_model("/lstm1_5.keras")


file_path = 'words_250000_train.txt'
words = load_data(file_path)
train(words)
