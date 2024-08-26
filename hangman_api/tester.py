import json
import requests
import random
import string
import secrets
import time
import re
import collections


import numpy as np
from keras.layers import Conv1D, LSTM, Dense, TimeDistributed, Embedding, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences#from keras.utils import pad_sequences

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.layers import Bidirectional
from keras.models import load_model



try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        self.incorrect_guesses = []

        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()

        self.current_dictionary = []

        self.max_word_length = 35
        self.letter_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                           's', 't',
                           'u', 'v', 'w', 'x', 'y', 'z']
        self.stats = ""
        self.record_length = []
        self.record_success = []

        self.probabilities = [0.0] * 26

        self.unigram, self.bigram, self.trigram, self.fourgram, self.fivegram = self.build_n_grams(self.full_dictionary)

        self.tries_remaining = 6

        self.model = load_model("lstm16.h5")

    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']

        data = {link: 0 for link in links}

        for link in links:

            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    """
    def train_data(self):
        X = []
        string = "abcdefghijklmnopqrstuvwxyz"
        letter_index = {s:i for i,s in enumerate(string)}
        for word in self.full_dictionary:

            N = len(word)
            X.append("_"*N)
            word_list = []
            for i in range(N):
    """

    def build_n_grams(self, dictionary):
        '''
        build nested dictionary containing occurences for n (1-5) sequences of letters
        unigrams and bigrams have an extra level for length of the word
        for unigram, take only unique letters within each word
        '''
        unigram = collections.defaultdict(lambda: collections.defaultdict(int))
        bi_gram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        tri_gram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        four_gram = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))))
        five_gram = collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))))

        # go through each word in the dictionary
        for word in dictionary:
            # check each letter in the dictionary and update the n-gram
            for i in range(len(word) - 4):
                bi_gram[len(word)][word[i]][word[i + 1]] += 1
                tri_gram[word[i]][word[i + 1]][word[i + 2]] += 1
                four_gram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]] += 1
                five_gram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]] += 1
            i = len(word) - 4

            # fill out the rest of the n-grams for words too short
            if len(word) == 2:
                bi_gram[len(word)][word[0]][word[1]] += 1
            elif len(word) == 3:
                bi_gram[len(word)][word[0]][word[1]] += 1
                bi_gram[len(word)][word[1]][word[2]] += 1
                tri_gram[word[0]][word[1]][word[2]] += 1

            # fill out rest of the (1-4)-grams
            elif len(word) >= 4:
                bi_gram[len(word)][word[i]][word[i + 1]] += 1
                bi_gram[len(word)][word[i + 1]][word[i + 2]] += 1
                bi_gram[len(word)][word[i + 2]][word[i + 3]] += 1
                tri_gram[word[i]][word[i + 1]][word[i + 2]] += 1
                tri_gram[word[i + 1]][word[i + 2]][word[i + 3]] += 1
                four_gram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]] += 1

            # fill out unigrams
            for letter in set(word):
                unigram[len(word)][letter] += 1

        return unigram, bi_gram, tri_gram, four_gram, five_gram

    def recalibrate_n_grams(self):
        '''
        works
        re-tabulates the n-grams after eliminating any incorrectly guessed letters
        updates the dictionary to remove words containing incorrectly guessed letters
        '''
        # updates the dictionary to remove words containing incorrectly guessed letters
        new_dict = [word for word in self.full_dictionary if not set(word).intersection(set(self.incorrect_guesses))]
        self.unigram, self.bigram, self.trigram, self.fourgram, self.fivegram = self.build_n_grams(new_dict)

    def fivegram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '0' if letter has not been guessed
        Flow: uses tri-gram to calculate the probability of a certain letter appearing in a five-letter sequence for a word of given length
        Output: probabilities for each letter to be used in next level
        '''
        len_chars = 26
        # vector of probabilities for each letter
        probs = [0.0] * len_chars

        total_count = 0
        letter_count = [0] * len_chars

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 4):

            # case 1: "letter letter letter letter blank"
            if word[i] != '0' and word[i + 1] != '0' and word[i + 2] != '0' and word[i + 3] != '0' and word[
                i + 4] == '0':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]

                # calculate occurences of "anchor_letter_1 anchor_letter_2 blank" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                        letter] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            letter]
                        letter_count[j] += \
                            self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter]

            # case 2: "letter letter letter blank letter"
            elif word[i] != '0' and word[i + 1] != '0' and word[i + 2] != '0' and word[i + 3] == '0' and word[
                i + 4] != '0':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 4]

                # calculate occurences of "anchor_letter_1 blank anchor_letter_2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][
                        anchor_letter_4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][
                            anchor_letter_4]
                        letter_count[j] += self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][
                            anchor_letter_4]

            # case 3: letter letter blank letter letter
            elif word[i] != '0' and word[i + 1] != '0' and word[i + 2] == '0' and word[i + 3] != '0' and word[
                i + 4] != '0':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 3]
                anchor_letter_4 = word[i + 4]

                # calculate occurences of "blank anchor_letter_1 anchor_letter_2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.fivegram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][
                        anchor_letter_4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][
                            anchor_letter_4]
                        letter_count[j] += self.fivegram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][
                            anchor_letter_4]

            # case 4: letter blank letter letter letter
            elif word[i] != '0' and word[i + 1] == '0' and word[i + 2] != '0' and word[i + 3] != '0' and word[
                i + 4] != '0':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]
                anchor_letter_4 = word[i + 4]

                # calculate occurences of "blank anchor_letter_1 anchor_letter_2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.fivegram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][
                        anchor_letter_4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][
                            anchor_letter_4]
                        letter_count[j] += self.fivegram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][
                            anchor_letter_4]

            # case 5: blank letter letter letter letter
            elif word[i] == '0' and word[i + 1] != '0' and word[i + 2] != '0' and word[i + 3] != '0' and word[
                i + 4] != '0':
                anchor_letter_1 = word[i + 1]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]
                anchor_letter_4 = word[i + 4]

                # calculate occurences of "blank anchor_letter_1 anchor_letter_2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.fivegram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][
                        anchor_letter_4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][
                            anchor_letter_4]
                        letter_count[j] += self.fivegram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][
                            anchor_letter_4]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_set)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.80)

        # run the next level down
        return self.fourgram_probs(word)

    def fourgram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '0' if letter has not been guessed
        Flow: uses tri-gram to calculate the probability of a certain letter appearing in a four-letter sequence for a word of given length
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0.0] * len(self.letter_set)

        total_count = 0
        letter_count = [0] * len(self.letter_set)

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 3):

            # case 1: "letter letter letter blank"
            if word[i] != '0' and word[i + 1] != '0' and word[i + 2] != '0' and word[i + 3] == '0':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]

                # calculate occurences of "anchor_letter_1 anchor_letter_2 blank" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.fourgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][
                        letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter]
                        letter_count[j] += self.fourgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter]

            # case 2: "letter letter blank letter"
            elif word[i] != '0' and word[i + 1] != '0' and word[i + 2] == '0' and word[i + 3] != '0':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 3]

                # calculate occurences of "anchor_letter_1 blank anchor_letter_2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.fourgram[anchor_letter_1][anchor_letter_2][letter][
                        anchor_letter_3] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3]
                        letter_count[j] += self.fourgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3]

            # case 3: letter blank letter letter
            elif word[i] != '0' and word[i + 1] == '0' and word[i + 2] != '0' and word[i + 3] != '0':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]

                # calculate occurences of "blank anchor_letter_1 anchor_letter_2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.fourgram[anchor_letter_1][letter][anchor_letter_2][
                        anchor_letter_3] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3]
                        letter_count[j] += self.fourgram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3]

            # case 4: blank letter letter letter
            elif word[i] == '0' and word[i + 1] != '0' and word[i + 2] != '0' and word[i + 3] != '0':
                anchor_letter_1 = word[i + 1]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]

                # calculate occurences of "blank anchor_letter_1 anchor_letter_2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.fourgram[letter][anchor_letter_1][anchor_letter_2][
                        anchor_letter_3] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3]
                        letter_count[j] += self.fourgram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_set)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.90)

        # run the next level down
        return self.trigram_probs(word)

    def trigram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '0' if letter has not been guessed
        Flow: uses tri-gram to calculate the probability of a certain letter appearing in a three-letter sequence for a word of given length
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0.0] * len(self.letter_set)

        total_count = 0
        letter_count = [0] * len(self.letter_set)

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 2):

            # case 1: "letter letter blank"
            if word[i] != '0' and word[i + 1] != '0' and word[i + 2] == '0':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]

                # calculate occurences of "anchor_letter_1 anchor_letter_2 blank" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.trigram[anchor_letter_1][anchor_letter_2][
                        letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.trigram[anchor_letter_1][anchor_letter_2][letter]
                        letter_count[j] += self.trigram[anchor_letter_1][anchor_letter_2][letter]

            # case 2: "letter blank letter"
            elif word[i] != '0' and word[i + 1] == '0' and word[i + 2] != '0':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 2]

                # calculate occurences of "anchor_letter_1 blank anchor_letter_2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.trigram[anchor_letter_1][letter][
                        anchor_letter_2] > 0 and letter not in self.guessed_letters:
                        total_count += self.trigram[anchor_letter_1][letter][anchor_letter_2]
                        letter_count[j] += self.trigram[anchor_letter_1][letter][anchor_letter_2]

            # case 3: blank letter letter
            elif word[i] == '0' and word[i + 1] != '0' and word[i + 2] != '0':
                anchor_letter_1 = word[i + 1]
                anchor_letter_2 = word[i + 2]

                # calculate occurences of "blank anchor_letter_1 anchor_letter_2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.trigram[letter][anchor_letter_1][
                        anchor_letter_2] > 0 and letter not in self.guessed_letters:
                        total_count += self.trigram[letter][anchor_letter_1][anchor_letter_2]
                        letter_count[j] += self.trigram[letter][anchor_letter_1][anchor_letter_2]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_set)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.40)

        # run the next level down
        return self.bigram_probs(word)

    def bigram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '0' if letter has not been guessed
        Flow: uses bi-gram to calculate the probability of a certain letter appearing in a two-letter sequence for a word of given length
              updates the probabilities set in trigram_probs
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0.0] * len(self.letter_set)

        total_count = 0
        letter_count = [0] * len(self.letter_set)

        # traverse the word and find either patterns of "letter blank" or "blank letter"
        for i in range(len(word) - 1):
            # case 1: "letter blank"
            if word[i] != '0' and word[i + 1] == '0':
                anchor_letter = word[i]

                # calculate occurences of "anchor_letter blank" and each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.bigram[len(word)][anchor_letter][letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.bigram[len(word)][anchor_letter][letter]
                        letter_count[j] += self.bigram[len(word)][anchor_letter][letter]

            # case 2: "blank letter"
            elif word[i] == '0' and word[i + 1] != '0':
                anchor_letter = word[i + 1]

                # calculate occurences of "blank anchor_letter" and each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.bigram[len(word)][letter][anchor_letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.bigram[len(word)][letter][anchor_letter]
                        letter_count[j] += self.bigram[len(word)][letter][anchor_letter]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_set)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.20)

        # return letter associated with highest probability

        return self.unigram_probs(word)

    def unigram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '0' if letter has not been guessed
        Flow: uses unigram to calculate the probability of a certain letter appearing in a any blank space
              updates the probabilities set in bigram_probs
        Output: letter with the overall highest probability

        '''

        # vector of probabilities for each letter
        probs = [0.0] * len(self.letter_set)

        total_count = 0
        letter_count = [0] * len(self.letter_set)

        # traverse the word and find blank spaces
        for i in range(len(word)):
            # case 1: "letter blank"
            if word[i] == '0':

                # calculate occurences of pattern and each letter not guessed yet
                for j, letter in enumerate(self.letter_set):
                    if self.unigram[len(word)][letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.unigram[len(word)][letter]
                        letter_count[j] += self.unigram[len(word)][letter]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_set)):
                probs[i] = letter_count[i] / total_count
        t = 0
        if len(self.guessed_letters) < 2:
            t = 1
        # interpolate probabilities
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.1) + probs[i] * t * 0.15

        # adjust probabilities so they sum to one (not necessary but looks better)
        final_probs = [0.0] * len(self.letter_set)
        if sum(self.probabilities) > 0:
            for i in range(len(self.probabilities)):
                final_probs[i] = self.probabilities[i] / sum(self.probabilities)

        self.probabilities = final_probs

        # find letter with largest probability
        max_prob = 0
        guess_letter = ''
        for i, letter in enumerate(self.letter_set):
            if self.probabilities[i] > max_prob:
                max_prob = self.probabilities[i]
                guess_letter = letter
        #print("ngram", guess_letter,len(self.incorrect_guesses))
        # if no letter chosen from above, pick a random one (extra weight on vowels)
        if guess_letter == '':
            letters = self.letter_set.copy()
            random.shuffle(letters)
            letters_shuffled = ['e', 'a', 'i', 'o', 'u'] + letters
            for letter in letters_shuffled:
                if letter not in self.guessed_letters:
                    return letter

        return guess_letter

    def predict_word(self, word):
        # print("LSTM")
        max_word_length = self.max_word_length
        guessed_letters = self.guessed_letters
        model = self.model
        Len = len(word)
        chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                 't',
                 'u', 'v', 'w', 'x', 'y', 'z']
        char_to_int = dict((c, i + 1) for i, c in enumerate(chars))
        int_to_char = dict((i + 1, c) for i, c in enumerate(chars))
        word_encoded = [char_to_int[char] if char in char_to_int else char_to_int['0'] for char in word]

        word_padded = pad_sequences([word_encoded], maxlen=max_word_length, padding='post')

        # incorrect_guess_vector = [29] * (len(chars)-1)
        # for w in self.incorrect_guesses:
        #    incorrect_guess_vector[char_to_int[w]-2] = 30

        prediction = model.predict(word_padded, verbose=0)
        predict_vector = prediction[0]
        #print(word)
        if Len<10 and Len<16:
            model = load_model("lstm11_15_seq_new.h5")
            prediction = model.predict(word_padded, verbose = 0)
            predict_vector2 = prediction[0]
        else:
            predict_vector2 = predict_vector
        probs = [0.0] * 28

        for i, char in enumerate(word):
            if char == '0':
                predict_vector[i][0] = predict_vector[i][1] = 0
                predict_vector2[i][0] = predict_vector2[i][1] = 0
                predict_vector[i] += predict_vector2[i]
                probabilities = predict_vector[i] / sum(predict_vector[i])
                probs += probabilities
        predicted_char = ""
        max_prob = 0
        probs[0] = 0.0
        probs[1] = 0.0
        probs = probs / sum(probs)
        temp = ""

        if self.tries_remaining < 2:
            self.recalibrate_n_grams()
            temp = self.fivegram_probs(word)
            probs[2:] += 1.2 * self.probabilities

        for i in range(2, 28):
            if probs[i] > max_prob and int_to_char[i] not in guessed_letters:
                max_prob = probs[i]
                predicted_char = int_to_char[i]

        return predicted_char  # Return None if all characters are guessed or no suitable character is found

    def guess(self, word):  # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_", "0")

        # find length of passed word
        len_word = len(clean_word)

        self.incorrect_guesses = list(set(self.guessed_letters) - set(word))

        # clear out probabilities from last guess
        self.probabilities = [0.0] * len(self.letter_set)




        # print(self.incorrect_guesses)

        dash_count = 0
        for w in clean_word:
            if w == "0": dash_count += 1
        if len_word < 6:
            self.model = load_model("lstm1_5_seq.h5")
        elif len_word < 11:
            self.model = load_model("lstm6_10_seq_new.h5")

        else:
            self.model = load_model("lstm16.h5")

        # to start, n gram
        if len_word < 7 and dash_count > int(0.4 * len_word) and len(self.incorrect_guesses) < 3:
            guess_letter = self.fivegram_probs(clean_word)
            self.recalibrate_n_grams()
        elif len_word >= 7 and dash_count > int(0.5 * len_word) and len(self.incorrect_guesses) < 2:
            guess_letter = self.fivegram_probs(clean_word)
            self.recalibrate_n_grams()
        else:
            guess_letter = self.predict_word(clean_word)

        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################

    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary

        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.incorrect_guesses = []
        self.probabilities = [0.0] * len(self.letter_set)
        self.recalibrate_n_grams()

        response = self.request("/new_game", {"practice": practice})
        if response.get('status') == "approved":
            game_id = response.get('game_id')
            word = response.get('word')
            self.tries_remaining = tries_remains = response.get('tries_remains')
            if verbose:
                print(
                    "Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id,
                                                                                                                tries_remains,
                                                                                                                word))
            while tries_remains > 0:
                # get guessed letter from user code
                guess_letter = self.guess(word)

                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))

                try:
                    res = self.request("/guess_letter",
                                       {"request": "guess_letter", "game_id": game_id, "letter": guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e

                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                t = tries_remains
                tries_remains = res.get('tries_remains')
                if t == tries_remains:
                    self.stats += "0"
                else:
                    self.stats += "1"

                if status == "success":
                    self.record_success.append(1)
                    self.record_length.append(len(word))
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status == "failed":
                    self.record_success.append(0)
                    self.record_length.append(len(word))
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status == "ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status == "success"

    def my_status(self):
        return self.request("/my_status", {})

    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result


class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)



api = HangmanAPI(access_token="bbcf518c454430a36f7a584c2c81c4", timeout=2000)

stats = ""
[old_practice_runs, total_recorded_runs, total_recorded_successes, old_practice_successes] = api.my_status()
for i in range(850):
    api.start_game(practice=0, verbose=False)
    api.stats += ","
    [new_practice_runs, total_recorded_runs, total_recorded_successes,
     new_practice_successes] = api.my_status()  # Get my game stats: (# of tries, # of wins)
    practice_success_rate = (total_recorded_successes) / (total_recorded_runs)

    print('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (
    total_recorded_runs, practice_success_rate))
    stats = api.stats
stats = stats.split(",")
