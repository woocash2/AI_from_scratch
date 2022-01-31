import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.mnist as mnist


def one_hot_label(digit):
    x = np.zeros(10)
    x[digit] = 1
    return x


def fun_with_tests(model, test_data, test_labels, test_features, test_reals, tries):
    for i in range(tries):
        index = random.randint(0, len(test_data))
        print('The number is:', model.decide_single(test_features[index]))
        plt.imshow(test_data[index])
        plt.show()


class Network:

    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.n = input_size
        self.m = hidden_size
        self.k = output_size
        self.lr = learning_rate

        self.W1 = np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        self.B1 = np.random.uniform(-0.1, 0.1, hidden_size)

        self.W2 = np.random.uniform(-0.1, 0.1, (hidden_size, output_size))
        self.B2 = np.random.uniform(-0.1, 0.1, output_size)

        self.H = None
        self.Hsig = None
        self.O = None
        self.Osig = None

    def sigmoid(self, matrix):
        return 1 / (1 + np.exp(-matrix))

    def predict(self, X):
        self.H = np.matmul(X, self.W1) + np.array([self.B1] * len(X))
        self.Hsig = self.sigmoid(self.H)
        self.O = np.matmul(self.Hsig, self.W2) + np.array([self.B2] * len(self.Hsig))
        self.Osig = self.sigmoid(self.O)
        return self.Osig

    def decide_single(self, features):
        batch = np.array(features)
        return np.argmax(self.predict(batch))

    def mean_grad_bias(self, grad):
        return 1 / len(grad) * np.sum(grad, axis=0)

    '''
    X --> X*W1 + B1 = H --> sig(H) = Hsig --> Hsig*W2 + B2 = O --> sig(O) = Osig = Y --> L  
    '''
    def fit(self, batch, reals):
        Y = self.predict(batch)
        D = (Y - reals)

        Yprim = np.multiply(Y, 1 - Y)
        Q = np.multiply(Yprim, D)

        grad_W2 = np.matmul(np.transpose(self.Hsig), Q)
        grad_B2 = self.mean_grad_bias(Q)

        Hprim = np.multiply(self.Hsig, 1 - self.Hsig)
        P = np.matmul(Q, np.transpose(self.W2))
        R = np.multiply(P, Hprim)

        grad_W1 = np.matmul(np.transpose(batch), R)
        grad_B1 = self.mean_grad_bias(R)

        self.W2 -= self.lr * grad_W2
        self.B2 -= self.lr * grad_B2
        self.W1 -= self.lr * grad_W1
        self.B1 -= self.lr * grad_B1

    def accuracy(self, batch, reals):
        pred = self.predict(batch)
        success = 0
        for i in range(len(pred)):
            if np.argmax(pred[i]) == np.argmax(reals[i]):
                success += 1
        return success / len(pred)


if __name__ == '__main__':

    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_reals = np.array([np.zeros(10)] * len(train_labels))
    for i in range(len(train_labels)):
        train_reals[i] = one_hot_label(train_labels[i])

    test_reals = np.array([np.zeros(10)] * len(test_labels))
    for i in range(len(test_labels)):
        test_reals[i] = one_hot_label(test_labels[i])

    train_features = np.array([np.zeros(len(train_data[0]) ** 2)] * len(train_data))
    for i in range(len(train_data)):
        train_features[i] = np.array(train_data[i]).flatten()
        train_features[i] = train_features[i] / np.linalg.norm(train_features[i])

    test_features = np.array([np.zeros(len(test_data[0]) ** 2)] * len(test_data))
    for i in range(len(test_data)):
        test_features[i] = np.array(test_data[i]).flatten()
        test_features[i] = test_features[i] / np.linalg.norm(test_features[i])

    epochs = 15
    learning_rate = 0.3
    batch_size = 10
    hidden_layer_size = 100

    model = Network(len(train_features[0]), hidden_layer_size, len(train_reals[0]), learning_rate)

    print(model.accuracy(test_features, test_reals))
    for e in range(epochs):
        for i in range(0, len(train_features), batch_size):
            batch = np.array(train_features[i:i + batch_size])
            reals = np.array(train_reals[i:i + batch_size])
            model.fit(batch, reals)
        print(e, model.accuracy(test_features, test_reals))

    fun_with_tests(model, test_data, test_labels, test_features, test_reals, 100)
