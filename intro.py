import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle



data = pd.read_csv("student_mat_2173a47420.csv", sep = ";")
print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
best = 0
for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    if accuracy > best:
        best = accuracy
        with open("student_grades.pickle", "wb") as f:
            pickle.dump(linear, f)


# load model
pickle_in = open("student_grades.pickle", "rb")
linear = pickle.load(pickle_in)

print("   -----")
print("accuracy", best)
print("co: ", linear.coef_)
print("Interecept:", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])



# Drawing and plotting model
plot = "G1"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("G1")
plt.show()
