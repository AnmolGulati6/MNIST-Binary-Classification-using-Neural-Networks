# MNIST Binary Classification using Neural Networks

This project is a binary classification task that involves training neural networks to classify handwritten digits from the MNIST dataset. The goal is to differentiate between two classes: 0 and 7. The project consists of two files, `p1.py` and `p2.py`, implementing logistic regression and a neural network with one hidden layer, respectively.

## Project Details

- Project Name: MNIST Binary Classification using Neural Networks
- Course: CS 540 - Intro to Artificial Intelligence
- University: UW Madison

## p1.py

### Description

The `p1.py` script implements logistic regression to classify the digits in the MNIST dataset as either 0 or 7. It performs the following steps:

1. Loads the MNIST training data.
2. Selects the samples with labels 0 and 7.
3. Preprocesses the data by normalizing the pixel values between 0 and 1.
4. Modifies the labels to be 0 for digit 0 and 1 for digit 7.
5. Configures hyperparameters such as the number of epochs and learning rate.
6. Initializes random weights and bias.
7. Trains the logistic regression model using gradient descent.
8. Prints the loss, loss reduction, and accuracy at each epoch.
9. Writes the results to a file.

### Usage

To run the `p1.py` script, ensure that you have the necessary dependencies installed. Then, execute the script using a Python interpreter:

```shell
python p1.py
```
## p2.py

### Description

The `p2.py` script trains a neural network with one hidden layer to classify the MNIST digits as either 0 or 7. It performs the following steps:

1. Loads the MNIST training data.
2. Selects the samples with labels 0 and 7.
3. Preprocesses the data by normalizing the pixel values between 0 and 1.
4. Modifies the labels to be 0 for digit 0 and 1 for digit 7.
5. Defines the sigmoid activation function and its derivative.
6. Configures hyperparameters such as the number of units in the hidden layer, learning rate, and number of epochs.
7. Initializes random weights and biases.
8. Trains the neural network using the backpropagation algorithm and stochastic gradient descent.
9. Evaluates the model's performance using the mean squared error (MSE) loss function and accuracy.
10. Writes the results to a file.

### Training the Model

The model is trained using the backpropagation algorithm and stochastic gradient descent. Weights (`w1` and `w2`) and biases (`b1` and `b2`) are randomly initialized within the range of -1 to 1. The training process involves iterating over the dataset for a specified number of epochs. In each epoch, the dataset is shuffled, and for each training example, the forward pass is performed to compute the network's output. The error is then propagated back through the network to update the weights and biases using gradient descent.

The loss function used is the mean squared error (MSE), which measures the difference between the predicted output and the actual label. The model's performance is evaluated using accuracy, calculated as the percentage of correctly classified instances.

### Usage

To run the `p2.py` script, ensure that you have the necessary dependencies installed. Then, execute the script using a Python interpreter:

```shell
python p2.py
```


