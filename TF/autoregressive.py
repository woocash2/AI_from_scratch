import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_autoregressive(length, coefs, var):
    autoreg = np.zeros(length)
    autoreg[0] = 0.1
    autoreg[1] = 0.1
    autoreg[2] = 0.1
    for i in range(3, length):
        autoreg[i] = np.random.uniform(0, var)
        for j in range(1, min(i + 1, 4)):
            autoreg[i] += autoreg[i - j] * coefs[j - 1]
    return autoreg


def fill_gaps(series):
    ser = []
    for i in range(len(series) - 1):
        ser.append(series[i])
        ser.append([(series[i][0] + series[i + 1][0]) / 2.0])
    ser.append(series[-1])
    return ser


def add_diffs(series):
    ser = []
    for i in range(len(series) - 1):
        ser.append([series[i + 1][0] - series[i][0]])
    series.extend(ser)
    print(series)


series = get_autoregressive(1200, [0.6, -0.5, -0.2], 0.1)
avgabs = 0


for x in series:
    for y in series:
        avgabs += np.abs(x - y)
avgabs /= 1200 * 1200

print('avg abs: ', avgabs)

domain = np.linspace(0, 100, 100)
plt.plot(domain, series[:100])
plt.show()


data = [[[series[j]] for j in range(i, i + 10)] for i in range(1000)]
targets = [[series[i + 10]] for i in range(1000)]
for i in range(len(data)):
    add_diffs(data[i])

print(data[0])

prob = 0.2

xtr = []
ytr = []
xte = []
yte = []

for i in range(1000):
    outcome = np.random.uniform(0, 1)
    if outcome < prob:
        xte.append(data[i])
        yte.append(targets[i])
    else:
        xtr.append(data[i])
        ytr.append(targets[i])

tr = list(zip(xtr, ytr))
np.random.shuffle(tr)
te = list(zip(xte, yte))
np.random.shuffle(te)

x_train = np.array([a[0] for a in tr], dtype=float)
y_train = np.array([a[1] for a in tr], dtype=float)
x_test = np.array([a[0] for a in te], dtype=float)
y_test = np.array([a[1] for a in te], dtype=float)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=64, return_sequences=True, activation='tanh', recurrent_dropout=0.2, dropout=0.1))
model.add(tf.keras.layers.LSTM(units=32, return_sequences=False, activation='tanh', dropout=0.2))
model.add(tf.keras.layers.Dense(units=1))

model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.RMSprop(), metrics=[tf.keras.metrics.mean_absolute_error])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

num = len(x_test)
domain = np.linspace(0, num, num)
plt.plot(domain, y_test)
y_pred = model.predict(x_test)
plt.plot(domain, y_pred)
plt.show()


for i in range(len(x_test)):
    x = x_test[i]
    y = y_test[i]
    print('x:', x)
    print()
    print('pred: ', model.predict(np.array([x])))
    print('true:', y)
    print()
    print()
    print()