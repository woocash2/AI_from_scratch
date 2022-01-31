import numpy as np
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
        print('The number is:', model.decide(test_features[index]))
        plt.imshow(test_data[index])
        plt.show()


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.layers = [np.array([])] * len(layer_sizes)
        self.weight_grads = [np.array([])] * (len(layer_sizes) - 1)
        self.bias_grads = [np.array([])] * (len(layer_sizes))

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.uniform(-0.1, 0.1, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.random.uniform(-0.1, 0.1, layer_sizes[i + 1]))
        self.weights.append(np.identity(layer_sizes[-1]))

    def sigmoid(self, matrix):
        return 1 / (1 + np.exp(-matrix))

    def sigmoid_prim(self, matrix):
        return np.multiply(matrix, 1 - matrix)

    def predict(self, batch):
        self.layers[0] = batch
        for i in range(len(self.layers) - 1):
            semi_layer = np.matmul(self.layers[i], self.weights[i]) + np.array([self.biases[i]] * len(batch))
            self.layers[i + 1] = self.sigmoid(semi_layer)
        return self.layers[-1]

    def decide(self, features):
        batch = np.array([features])
        return np.argmax(self.predict(batch))

    def column_mean(self, matrix):
        return np.sum(matrix, axis=0) / len(matrix)

    def fit(self, batch, reals):
        self.predict(batch)
        self.bias_grads[-1] = self.layers[-1] - reals

        for i in range(len(self.layers) - 2, -1, -1):
            self.bias_grads[i] = np.matmul(self.bias_grads[i + 1], np.transpose(self.weights[i + 1]))
            self.bias_grads[i] = np.multiply(self.bias_grads[i], self.sigmoid_prim(self.layers[i + 1]))
            self.weight_grads[i] = np.matmul(np.transpose(self.layers[i]), self.bias_grads[i])

        for i in range(len(self.layers) - 1):
            self.weights[i] -= self.learning_rate * self.weight_grads[i]
            self.biases[i] -= self.learning_rate * self.column_mean(self.bias_grads[i])

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

    epochs = 5
    learning_rate = 0.5
    batch_size = 20
    sizes = [len(train_features[0]), 64, 32, len(train_reals[0])]

    model = NeuralNetwork(sizes, learning_rate)

    print(model.accuracy(test_features, test_reals))
    for e in range(epochs):
        if e % 10 == 9:
            model.learning_rate /= 2.0
        for i in range(0, len(train_features), batch_size):
            batch = np.array(train_features[i:i + batch_size])
            reals = np.array(train_reals[i:i + batch_size])
            model.fit(batch, reals)
        print(e, model.accuracy(test_features, test_reals))

    fun_with_tests(model, test_data, test_labels, test_features, test_reals, 100)
