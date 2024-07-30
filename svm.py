import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)

x = cancer.data  # All of the features
y = cancer.target  # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train[:5], y_train[:5])
clf = svm.SVC(kernel = "linear", C=1)

model = KNeighborsClassifier(n_neighbors=13)
model.fit(x_train, y_train)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test) # Predict values for our test data
y_pred2 = model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred) # Test them against our correct values
acc2 = metrics.accuracy_score(y_test, y_pred2)
print(acc, acc2)