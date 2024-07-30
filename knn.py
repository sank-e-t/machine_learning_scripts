import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

data = pd.read_csv("car.data")
print(data.head())
le = preprocessing.LabelEncoder()
data1 = data
for i in list(data.columns):
    data1[i] = le.fit_transform(list(data[i]))

print(data1.head())
X = np.array(data1.drop(["class"], 1))
Y = np.array(data1["class"])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
best = 0
for i in range(4,11):
    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    if acc > best:
        best = acc
        with open("car_class.pickle", "wb") as f:
            print(i)
            pickle.dump(model, f)


pickle_in = open("car_class.pickle", "rb")
model = pickle.load(pickle_in)
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 7, True)
    print("Neighbors: ", n)

