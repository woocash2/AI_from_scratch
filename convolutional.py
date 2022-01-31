import numpy as np


class Convolutional:
    def __init__(self, conv_sizes, dense_sizes, input_shape, kernel_size=5):
        self.conv_layers = [np.array([])] * (len(conv_sizes) + 1)
        self.dense_layers = [np.array([])] * (len(dense_sizes) + 1)

        self.kernels = []
        for i in range(int(len(conv_sizes))):
            self.kernels.append([])
            for j in range(conv_sizes[i]):
                self.kernels[i].append(np.random.normal(0.0, 0.1, size=(kernel_size, kernel_size)))

        self.weights = []
        self.biases = []
        self.flatten_size = input_shape[0] * input_shape[1]

        self.weights.append(np.random.normal(0.0, 0.1, size=(self.flatten_size, dense_sizes[0])))
        self.biases.append(np.random.normal(0.0, 0.1, size=dense_sizes[0]))

        for i in range(1, int(len(dense_sizes))):
            self.weights.append(np.random.normal(0.0, 0.1, size=(dense_sizes[i - 1], dense_sizes[i])))
            self.biases.append(np.random.normal(0.0, 0.1, size=dense_sizes[i]))
        self.weights.append(np.identity(dense_sizes[-1]))


    def predict(self, x):


c = Convolutional([3, 4], [2, 3], (10, 10), 5)
print(c.weights)