
#!/usr/bin/env python
# coding: utf-8

# # Trexquant Interview Project (The Hangman Game)
# 
# * Copyright Trexquant Investment LP. All Rights Reserved. 
# * Redistribution of this question without written consent from Trexquant is prohibited

# ## Instruction:
# For this coding test, your mission is to write an algorithm that plays the game of Hangman through our API server. 
# 
# When a user plays Hangman, the server first selects a secret word at random from a list. The server then returns a row of underscores (space separated)—one for each letter in the secret word—and asks the user to guess a letter. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. The user keeps guessing letters until either (1) the user has correctly guessed all the letters in the word
# or (2) the user has made six incorrect guesses.
# 
# You are required to write a "guess" function that takes current word (with underscores) as input and returns a guess letter. You will use the API codes below to play 1,000 Hangman games. You have the opportunity to practice before you want to start recording your game results.
# 
# Your algorithm is permitted to use a training set of approximately 250,000 dictionary words. Your algorithm will be tested on an entirely disjoint set of 250,000 dictionary words. Please note that this means the words that you will ultimately be tested on do NOT appear in the dictionary that you are given. You are not permitted to use any dictionary other than the training dictionary we provided. This requirement will be strictly enforced by code review.
# 
# You are provided with a basic, working algorithm. This algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearence that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary.
# 
# This benchmark strategy is successful approximately 18% of the time. Your task is to design an algorithm that significantly outperforms this benchmark.

# In[23]:


import json
import requests
import random
import string
import secrets
import time
import re
import collections

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


# In[36]:


class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        self.current_dictionary = []
        
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

    def guess_first(self, word): # word input example: "_ _ _ _ _ "
        ###############################################
        # just for the first letter, i think the original algorithm is quite alright
        ###############################################

        # find length of passed word
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []
        
        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break            
        
        return guess_letter



    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_",".")
        
        # find length of passed word
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []
        
        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break            
        
        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
        
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


# # API Usage Examples

# ## To start a new game:
# 1. Make sure you have implemented your own "guess" method.
# 2. Use the access_token that we sent you to create your HangmanAPI object. 
# 3. Start a game by calling "start_game" method.
# 4. If you wish to test your function without being recorded, set "practice" parameter to 1.
# 5. Note: You have a rate limit of 20 new games per minute. DO NOT start more than 20 new games within one minute.

# In[38]:


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



def simulate_missing_letters(word, char_to_int, p_missing=0.4):
    """
    Randomly replace letters with '0' to simulate missing letters,
    with p_missing chance for each letter.
    """
    rand_dic = {}
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    eliminated = list(set(chars) - set(word))

    for i in word:
        rand_dic[i] = rand_dic.get(i, random.random())
    simulated_word = ''
    for char in word:
        simulated_word += '0' if rand_dic[char] < p_missing else char
    return simulated_word


def preprocess_data(words, max_word_length, p_missing=0.4):
    sequences = []
    targets = []
    guessed = []
    chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    for _ in range(3):
        for word in words:
            #simulating eliminated guesses
            eliminated = list(set(chars) - set(word))
            random.shuffle(eliminated)

            s = ""
            rand_len = int(random.random() * 3)
            for i in eliminated[:rand_len]:
                s += i
            guessed.append([char_to_int[char] if char in char_to_int else char_to_int['0'] for char in s])
            #----------------------------------------------
            word_with_missing = simulate_missing_letters(word, char_to_int, p_missing)
            sequences.append([char_to_int[char] if char in char_to_int else char_to_int['0'] for char in word_with_missing])
            targets.append([char_to_int[char] for char in word])

    sequences = pad_sequences(sequences, maxlen=max_word_length, padding='post')
    guessed = pad_sequences(guessed, maxlen=6, padding='post')
    targets = pad_sequences(targets, maxlen=max_word_length+6, padding='post')

    sequences = np.concatenate((sequences, guessed), axis=1)
    return sequences, targets


def train(words):
    #words = api.full_dictionary
    max_word_length = max(len(word) for word in words)  # finding largest word, maybe switch it to 50 or something
    random.shuffle(words)  # removing the alphabetic order so that the model trains impartially

    words1_5 = []
    words6_10 = []
    words11_15 = []
    words16 = []
    for word in words:
        if len(word) < 6:
            words1_5.append(word)
        elif len(word) < 11 and len(word) > 5:
            words6_10.append(word)
        elif len(word) < 16 and len(word) > 10:
            words11_15.append(word)
        else:
            words16.append(word)

    chars = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    p_missing = 0.7
    # -----------------------------------------------------------------------------------------
    X, y = preprocess_data(words1_5, 5, p_missing)
    y = to_categorical(y, num_classes=len(chars))
    bs = 512
    epoch = 25
    model = Sequential()
    model.add(Embedding(input_dim=len(chars), output_dim=128, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.3, epochs=epoch, batch_size=bs,
              callbacks=[early_stopping])
    model.save("lstm1_5_hi_intense.h5")
    # -----------------------------------------------------------------------------------------
    X, y = preprocess_data(words6_10, 10, p_missing)
    y = to_categorical(y, num_classes=len(chars))
    bs = 2*bs
    model = Sequential()
    model.add(Embedding(input_dim=len(chars), output_dim=128, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.2, epochs= epoch, batch_size=bs,
              callbacks=[early_stopping])
    model.save("lstm6_10_hi_intense.h5")
    # -----------------------------------------------------------------------------------------
    X, y = preprocess_data(words11_15, 15, p_missing)
    y = to_categorical(y, num_classes=len(chars))

    model = Sequential()
    model.add(Embedding(input_dim=len(chars), output_dim=128, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.2, epochs=epoch, batch_size=bs,
              callbacks=[early_stopping])
    model.save("lstm11_15_hi_intense.h5")
    # -----------------------------------------------------------------------------------------
    X, y = preprocess_data(words16,  max_word_length, p_missing)
    y = to_categorical(y, num_classes=len(chars))
    bs = int(bs/2)
    model = Sequential()
    model.add(Embedding(input_dim=len(chars), output_dim=128, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

    model.fit(X, np.array(y), validation_split=0.2, epochs= epoch, batch_size=bs,
              callbacks=[early_stopping])
    model.save("lstm16_hi_intense.h5")

api = HangmanAPI(access_token="bbcf518c454430a36f7a584c2c81c4", timeout=2000)
words = api.full_dictionary
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

""""
word_with_missing = "an000"

guessed_letters = set(['e', 'o', 'i', 's', 't', 'd', 'a', "g"])
model = load_model('lstm1_5_intense.h5')
max_word_length = 5
predicted_char = predict_word(model, word_with_missing, max_word_length, guessed_letters)
print(predicted_char)

"""