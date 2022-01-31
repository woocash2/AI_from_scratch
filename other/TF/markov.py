import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_markov(num, coefs, var):
    randoms = np.random.normal(0.0, 1.0, num)
    markov = np.zeros(num)
    for i in range(num):
        markov[i] = randoms[i]
        for j in range(len(coefs)):
            markov[i] += coefs[j] * randoms[i - 1 - j]
    return markov


m = get_markov(10200, [5.0, -1.0, -1.0, -1.0, -1.0], 1.0)
domain = np.linspace(0, 10000, 10000)
plt.plot(domain, m[:10000])
plt.show()

data = [([[m[j]] for j in range(i, i + 10)], m[i + 10]) for i in range(1000)]
tr = []
te = []
prob = 0.25
print(data[0])
x = input()

for d in data:
    rv = np.random.uniform(0.0, 1.0)
    if rv > prob:
        tr.append(d)
    else:
        te.append(d)

print(len(tr))
print(len(te))
np.random.shuffle(tr)
np.random.shuffle(te)

x_tr = np.array([a[0] for a in tr])
y_tr = np.array([a[1] for a in tr])
x_te = np.array([a[0] for a in te])
y_te = np.array([a[1] for a in te])

print(x_tr[0])
print(y_tr[0])
print(x_te[0])
print(y_te[0])

rnn = tf.keras.models.Sequential()
rnn.add(tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=True, recurrent_dropout=0.2, dropout=0.1))
rnn.add(tf.keras.layers.LSTM(units=32, activation='tanh', return_sequences=False, dropout=0.2))
rnn.add(tf.keras.layers.Dense(units=1))

rnn.compile(optimizer='rmsprop', loss='mean_absolute_error', metrics='mean_absolute_error')
rnn.fit(x=x_tr, y=y_tr, validation_data=(x_te, y_te), epochs=100)

num = len(x_te)
domain = np.linspace(0, num, num)
plt.plot(domain, y_te)
y_pred = rnn.predict(x_te)
plt.plot(domain, y_pred)
plt.show()

for t in te:
    print('pred:', rnn.predict(np.array([t[0]])))
    print('true:', t[1])
    print()