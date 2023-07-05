# Written by Anmol Gulati

import numpy as np
import pandas as pd


# Part 1: Setup the data
def data_loader(file):
    df = pd.read_csv(file)
    x = (df.iloc[:, 1:] / 255.0).to_numpy()
    y = df.iloc[:, 0].to_numpy()
    return (x, y)


# load the training data
x_train, y_train = data_loader("mnist_train.csv")

# test_labels might be different for you
# 0 (label it 0) and 7 (label it 1)
test_labels = [0, 7]
indices = np.where(np.isin(y_train, test_labels))[0]

# get the indices of the training data that have labels 0 and 7
x = x_train[indices]
y = y_train[indices]

# label 0 as 0 and label 7 as 1
y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1


def sigmoid(x):
    ''' sigmoid function '''
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(o):
    ''' derivative of sigmoid function '''
    return o * (1 - o)


# Part 2: Configure the Hyperparameters
## number of units in the hidden layer
h = 28

# total number of pixels in an image (28x28)
# number of units in the input layer, i.e., 784
m = x.shape[1]

# adjust alpha and number of epochs by yourself
alpha = 0.001
num_epochs = 50
num_train = len(y)


def nnet(train_x, train_y, alpha, num_epochs, num_train):
    # initialize the weights and biases
    # (Important: see the dimensinos of the weights and biases)
    w1 = np.random.uniform(low=-1, high=1, size=(m, h))
    w2 = np.random.uniform(low=-1, high=1, size=(h, 1))
    b1 = np.random.uniform(low=-1, high=1, size=(h, 1))
    b2 = np.random.uniform(low=-1, high=1, size=(1, 1))

    # set a large number as the initial cost to be compared with in the 1st iteration
    loss_previous = 10e10

    for epoch in range(1, num_epochs + 1):
        # shuffle the dataset
        train_index = np.arange(num_train)
        np.random.shuffle(train_index)

        for i in train_index:
            # a1 will be of the dimension of 28 * 1
            a1 = sigmoid(w1.T @ train_x[i, :].reshape(-1, 1) + b1)
            # a2 is a 1*1 matrix
            a2 = sigmoid(w2.T @ a1 + b2)

            # dCdw1 will be a 28 * 784 matrix
            dCdw1 = (
                    (a2 - train_y[i])
                    * sigmoid_derivative(a2)
                    * w2
                    * sigmoid_derivative(a1)
                    * (train_x[i, :].reshape(1, -1))
            )

            # dCdb1 will be a 28 * 1 matrix
            dCdb1 = (
                    (a2 - train_y[i]) * sigmoid_derivative(a2) * w2 * sigmoid_derivative(a1)
            )

            # dCdw2 will be a a 28 * 1 matrix
            dCdw2 = (a2 - train_y[i]) * sigmoid_derivative(a2) * a1

            # dCdb2 will be a 1*1 matrix
            dCdb2 = (a2 - train_y[i]) * sigmoid_derivative(a2)

            # update w1, b1, w2, b2
            w1 = w1 - alpha * dCdw1.T
            b1 = b1 - alpha * dCdb1
            w2 = w2 - alpha * dCdw2
            b2 = b2 - alpha * dCdb2

        # train_x @ w1 will be num_train * 28 matrix
        # the output of the hidden layer will be a num_train * 28 matrix
        out_h = sigmoid(train_x @ w1 + b1.T)
        # the output of the output layer will be a num_train * 1 matrix
        out_o = sigmoid(out_h @ w2 + b2)

        loss = 0.5 * np.sum(np.square(y.reshape(-1, 1) - out_o))
        loss_reduction = loss_previous - loss
        loss_previous = loss
        correct = sum((out_o > 0.5).astype(int) == y.reshape(-1, 1))
        accuracy = (correct / num_train)[0]
        print(
            "epoch = ",
            epoch,
            " loss = {:.7}".format(loss),
            " loss reduction = {:.7}".format(loss_reduction),
            " correctly classified = {:.4%}".format(accuracy),
        )

        # You can apply your stop rule here if you would like

    return w1, b1, w2, b2


w1, b1, w2, b2 = nnet(x, y, alpha, num_epochs, num_train)

# Part 4: Write
file = 'result2.txt'
import os
from pathlib import Path

my_file = Path(file)
if my_file.is_file():
    os.remove(file)

# Q5
# first layer weights and biases
w1plusb1 = np.concatenate((w1, b1.T), axis=0)
w1plusb1.shape

f = open(file, 'a')
f.write('##5: \n')
for row in range(w1plusb1.shape[0]):
    for i in range(h):
        if i != 0:
            f.write(',')
        f.write('%.4f' % w1plusb1[row, i])
    f.write('\n')
f.close()

# Q6
# second layer weights and biases
w2plusb2 = np.concatenate((w2, b2), axis=0)
w2plusb2.shape

f = open(file, 'a')
f.write('##6: \n')
for i in range(w2plusb2.shape[0]):
    if i != 0:
        f.write(',')
    f.write('%.4f' % w2plusb2[i, 0])
f.write('\n')
f.close()

# Q7
# load the test set
x = np.loadtxt('test1.txt', delimiter=',')
x = x / 255.0

# output layer activations on the test set
# x @ w1 will be num_train * 28 matrix
# the output below will be a num_train * 28 matrix
a = sigmoid(x @ w1 + b1.T)
# the output below will be a num_train * 1 matrix
a = sigmoid(a @ w2 + b2)

f = open(file, 'a')
f.write('##7: \n')
for i in range(a.shape[0]):
    if i != 0:
        f.write(',')
    f.write('%.2f' % a[i, 0])
f.write('\n')
f.close()

# Q8 predictions
f = open(file, 'a')
f.write('##8: \n')
for i in range(a.shape[0]):
    if i != 0:
        f.write(',')
    predicted_value = (a[i, 0] > .5).astype(int)
    f.write('%.0f' % predicted_value)
f.write('\n')
f.close()

# Q9 find the first misclassified image
misclassified_image = None

for i in range(a.shape[0]):
    predicted_value = (a[i, 0] > .5).astype(int)
    if i >= 100 and predicted_value == 0:
        misclassified_image = i
        break

f = open(file, 'a')
f.write('##9: \n')
for i in range(x[misclassified_image, :].reshape(-1, 1).shape[0]):
    if i != 0:
        f.write(',')
    f.write('%.2f' % x[misclassified_image, :].reshape(-1, 1)[i, 0])
f.write('\n')
f.close()
