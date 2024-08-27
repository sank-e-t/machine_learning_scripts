hangman_test.py is a script one can use to load the model and manually test against the words with missing letters.
hm_train_split.py is the training script for the model.
tester.py makes the hangmanAI play the game.
The test set and the training set are disjoint.
The hangman solver at first uses N-gram model to fill in the blanks and when there is enough sequence to go around, it switches to the Bi-LSTM model.
In order to identify features more accurately, I created separate models for smaller words and larger words.
