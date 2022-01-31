import tensorflow as tf
from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential(layers=[
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=80, activation='relu'),
    tf.keras.layers.Dense(units=40, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=3, validation_data=(X_test, Y_test))
