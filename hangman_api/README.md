The hangman solver at first uses N-gram model to fill in the blanks and when there is enough sequence to go around, it switches to the Bi-LSTM model.
In order to identify features more accurately, I created separate models for smaller words and larger words.
The test set and the training set are disjoint.
The model currently has 53% winning rate with 6 lives. I need to add comments and tidy up the scripts in the near future. I might also try to make a deep reinforced learning algorithm to solve this problem to get a higher victory rate.

hangman_test.py is a script one can use to load the model and manually test against the words with missing letters.

hm_train_split.py is the training script for the model.

tester.py makes the hangmanAI play the game.

