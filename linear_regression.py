import numpy as np
import matplotlib.pyplot as plt

'''
Perceptron solving multidimensional linear regression.
2D and 3D examples.
'''


'''
Perceptron class.
To specify: n = dimensionality, lr = learning rate.
'''


class Perceptron:
    def __init__(self, n, lr):
        self.n = n
        self.w = np.random.uniform(-0.1, 0.1, n)
        self.b = np.random.uniform(-0.1, 0.1)
        self.lr = lr

    def predict(self, features):
        return np.dot(self.w, features) + self.b

    def fit(self, features, real):
        pred = self.predict(features)
        grad_w = (1 / self.n) * (pred - real) * features
        grad_b = (1 / self.n) * (pred - real)
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b


'''
2D example.
line_A and line_B define a straight line. For x = 0,...,99 values are chosen from the line + random noise from uniform
distribution. Then a perceptron is taught to establish a line which fits these points the best.
Results are being displayed on the plot.
'''
line_A = -5
line_B = 200

x = np.linspace(0, 99, num=100)
r = np.random.uniform(-10, 10, 100)
y = line_A * (x + r) + line_B

learning_rate = 0.0001
epochs = 5000

perceptron = Perceptron(1, learning_rate)
width = 1

for i in range(epochs):
    if i % 1000 == 0:
        width += 0.1
        plt.plot(perceptron.w * x + perceptron.b, c='green', linewidth=width)
    for j in range(100):
        perceptron.fit(np.array([x[j]]), y[j])


print(perceptron.w, perceptron.b)

plt.scatter(x, y, c='blue')
plt.plot(perceptron.w * x + perceptron.b, c='red', linewidth=2)
plt.show()
plt.clf()


'''
3D example. Similarly as previously a plane is chosen and points (i, j): i, j in {0,...,99} are given a value from
the plane + random uniform noise. Then perceptron looks for a plane which fits these points.
'''

plane_A = 5
plane_B = 2
plane_C = 30

X = np.linspace(0, 99, num=100)
Y = np.linspace(0, 99, num=100)
Z = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        Z[i][j] = plane_A * i + plane_B * j + plane_C + np.random.uniform(-20, 20)

learning_rate = 0.0001
epochs = 500
network = Perceptron(2, learning_rate)

for e in range(epochs):
    if e % 100 == 0:
        print("epoch", e)
    for i in range(100):
        for j in range(100):
            network.fit(np.array([X[i], Y[j]]), Z[i][j])

print(network.w, network.b)
