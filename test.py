import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier


def csv_reader(filename):
    # reader cvs files and store data and label into data and label
    data = []
    label = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([float(a) for a in row[:-1]])
            label.append(int(row[-1]))
    return np.array(data), np.array(label)


(data_train, label_train) = csv_reader('wine_train.csv')
(data_test, label_test) = csv_reader('wine_test.csv')
length = len(data_train)
print("mean for train:", np.mean(data_train, axis=0))
print("standard deviation for train:", np.std(data_train, axis=0))
# print("mean for test:", np.mean(data_test, axis=0))
# print("standard deviation for test:", np.std(data_test, axis=0))
data_train_2 = data_train[:, 0:2]
data_test_2 = data_test[:, 0:2]

# normalize the dataset
std = StandardScaler()
std.fit(data_train)
std_data_train = std.transform(data_train)
std_data_test = std.transform(data_test)

# extract only the first two features
# std_data_train_2 = np.hstack((std_data_train[:, 0].reshape(length, 1), std_data_train[:, 1].reshape(length, 1)))
# std_data_test_2 = np.hstack((std_data_test[:, 0].reshape(length, 1), std_data_test[:, 1].reshape(length, 1)))
std_data_train_2 = std_data_train[:, 0:2]
std_data_test_2 = std_data_test[:, 0:2]

# Perceptron Classifier
clf = Perceptron()  # linear model
clf.fit(std_data_train_2, label_train)  # fit data to model
print("w for first 2 features:", clf.coef_)
print("w0 for first 2 features:", clf.intercept_)
# predict training labels
pred_label_train_2 = clf.predict(std_data_train_2)
pred_label_test_2 = clf.predict(std_data_test_2)
# calculate the accuracy
print("accuracy for train_2:", accuracy_score(label_train, pred_label_train_2))
print("accuracy for test_2:", accuracy_score(label_test, pred_label_test_2))

clf.fit(std_data_train, label_train)
print("w for 13 features:", clf.coef_)
print("w0 for 13 features:", clf.intercept_)
pred_label_train = clf.predict(std_data_train)
pred_label_test = clf.predict(std_data_test)
print("accuracy for train_13:", accuracy_score(label_train, pred_label_train))
print("accuracy for test_13:", accuracy_score(label_test, pred_label_test))
print("--------------------------------------------------------------------------------------------")


# randomly choose starting weight vector and find the highest accuracy among 100 times

def best_acc(train, label1, test, label2, num_feature):
    acc_train = []
    acc_test = []
    w_train = []
    w0_train = []
    w_test = []
    w0_test = []
    for i in range(0, 100):
        init_w0 = np.random.randn(3)
        init_w = np.random.randn(3, num_feature)
        clf.fit(train, label1, coef_init=init_w, intercept_init=init_w0)
        p_label_train = clf.predict(train)
        p_label_test = clf.predict(test)
        acc_train.append(accuracy_score(label1, p_label_train))
        acc_test.append(accuracy_score(label2, p_label_test))
        w_train.append(clf.coef_)
        w0_train.append(clf.intercept_)

    print("accuracy for train:", acc_train[acc_train.index(max(acc_train))])
    print("final w for train:", w_train[acc_train.index(max(acc_train))])
    print("final w0 for train:", w0_train[acc_train.index(max(acc_train))])
    print("accuracy for test:", acc_test[acc_train.index(max(acc_train))])


print("100times")
print("For the first two features:")
best_acc(std_data_train_2, label_train, std_data_test_2, label_test, 2)
print("-----------------------------------------------------------------------------------------")
print("For the 13 features")
best_acc(std_data_train, label_train, std_data_test, label_test, 13)
print("-----------------------------------------------------------------------------------------")


class MSE_binary(LinearRegression):
    def __init__(self):
        super(MSE_binary, self).__init__()

    def predict(self, X):
        thr = 0.5  # may varying depending on Xw=b
        y = self._decision_function(X)
        label = [int(i + thr) for i in y]
        return label


def mse(train, label1, test, label2):
    binary_model = MSE_binary()
    mc_model = OneVsRestClassifier(binary_model)
    mc_model.fit(train, label1)
    p_label = mc_model.predict(test)
    acc = accuracy_score(label2, p_label)
    return acc


print("For pseudoinverse classifier:")
print("normalized data:")
print("the accuracy of 2 features:", mse(std_data_train_2, label_train, std_data_test_2, label_test))
print("the accuracy of 13 features:", mse(std_data_train, label_train, std_data_test, label_test))
print("unnormalized data:")
print("the accuracy of 2 features:", mse(data_train_2, label_train, data_test_2, label_test))
print("the accuracy of 13 features:", mse(data_train, label_train, data_test, label_test))
