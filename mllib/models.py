import numpy as np


def mse(X, Y):
    return ((X - Y) ** 2) / X.size


def dmse(X, Y):
    return 2. * (X - Y) / X.size


def crossentropy(X, Y):
    return np.multiply(-Y / X.size, np.log2(X))


def dcrossentropy(X, Y):
    return -(Y / X) / X.size


class Sequential:
    def __init__(self, loss='mse'):
        self.layers = []
        self.ls = loss
        self.inputs = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def loss(self, X, Y):
        return crossentropy(X, Y) if self.ls == 'crossentropy' else mse(X, Y)

    def dloss(self, X, Y):
        return dcrossentropy(X, Y) if self.ls == 'crossentropy' else dmse(X, Y)

    def predict(self, X):
        self.inputs.clear()
        self.inputs.append(X.copy())
        for layer in self.layers:
            Y = layer.forward(self.inputs[-1])
            self.inputs.append(Y)
        return self.inputs[-1]

    def backward(self, Y, lr):
        X = self.inputs[-1]
        D = self.dloss(X, Y)

        for i, L in reversed(list(enumerate(self.layers))):
            D = L.update_params_and_chain(D, lr)

    def summary_loss(self, data, labels):
        batches = float(len(data))
        ls = 0.0
        for X, Y in list(zip(data, labels)):
            ls += np.sum(self.loss(self.predict(X), Y) / batches)
        return ls

    def fit(self, data, labels, epochs, lr):
        for e in range(epochs):
            print('Epoch', e + 1, end=' ')
            for X, Y in list(zip(data, labels)):
                self.predict(X)
                self.backward(Y, lr)
            print('Loss:', self.summary_loss(data, labels))
        print('Finished')
