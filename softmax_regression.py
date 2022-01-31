import numpy as np
import random
import os
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


class SoftmaxModel:

    def __init__(self, input_size, output_size, learning_rate):
        self.inp_len = input_size
        self.out_len = output_size
        self.lr = learning_rate
        self.Weights = np.random.uniform(-0.1, 0.1, (input_size, output_size))
        self.bias = np.random.uniform(-0.1, 0.1, output_size)

    def softmax(self, O):
        O_exp = np.exp(O)
        row_sums = np.sum(O_exp, axis=1)
        O_exp = np.transpose(O_exp)
        O_exp = O_exp / row_sums
        return np.transpose(O_exp)

    def predict(self, batch):
        o = np.matmul(batch, self.Weights) + np.array([self.bias] * len(batch))
        o = self.softmax(o)
        return o

    def decide_single(self, features):
        batch = np.array(features)
        return np.argmax(self.predict(batch))

    def grad_weights(self, batch, real, predicted):
        return np.matmul(np.transpose(batch), predicted - real)

    def grad_bias(self, real, predicted):
        return np.sum(predicted - real, axis=0)

    def fit(self, batch, reals):
        pred = self.predict(batch)
        x = self.grad_weights(batch, reals, pred)
        self.Weights -= self.lr * self.grad_weights(batch, reals, pred)
        self.bias -= self.lr * self.grad_bias(reals, pred)

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

    epochs = 100
    learning_rate = 0.01
    batch_size = 10

    model = SoftmaxModel(len(train_features[0]), len(train_reals[0]), learning_rate)

    print(model.accuracy(test_features, test_reals))
    for e in range(epochs):
        for i in range(0, len(train_features), batch_size):
            batch = np.array(train_features[i:i + batch_size])
            reals = np.array(train_reals[i:i + batch_size])
            model.fit(batch, reals)
        print(model.accuracy(test_features, test_reals))

    fun_with_tests(model, test_data, test_labels, test_features, test_reals, 100)
