import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

print(tf.__version__)

'''
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file('aclImdb_v1', url, untar=True, cache_dir='.', cache_subdir='')

print(dataset)
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb_v1')
print(dataset_dir)
'''
dataset_dir = 'aclImdb'

print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'train')

print(os.listdir(train_dir))

sample_file = os.path.join(train_dir, 'pos', '0_9.txt')
with open(sample_file, 'r') as file:
    print(file.read())
'''
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
'''

batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)


for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print('Review', text_batch.numpy()[i])
        print('Label', label_batch.numpy()[i])

print('Label 0:', raw_train_ds.class_names[0])


raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)


raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size
)


def custom_standatization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', '')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


max_features = 10000
seq_length = 250

vectorize_layer = TextVectorization(
    standardize=custom_standatization,
    max_tokens=max_features,
    output_sequence_length=seq_length,
    output_mode='int'
)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label



text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print('review:', first_review)
print('label:', first_label)
print('vectorized:', vectorize_text(first_review, first_label))

print('1270 -->', vectorize_layer.get_vocabulary()[1270])


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


embedding_dim = 16

model = tf.keras.models.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])

print(model.summary())

model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
)

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

loss, accuracy = model.evaluate(test_ds)
print('Test accuracy:', accuracy)


history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


export_model = tf.keras.models.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy']
)

loss, accuracy = export_model.evaluate(raw_test_ds)
print('Raw test accuracy:', accuracy)


examples = [
    'The movie was amazing. It was a pleasure watching it.',
    'This piece of shit seems just wrong.',
    'Cool movie, would watch again.',
    'Bad, bad, bad, bad, bad.',
    'Cool, cool, cool, really cool.',
    'I am angry after watching the movie.',
    'Movie was not bad.',
    'Movie was not good.'
]

labels = [1, 0, 1, 0, 1, 0]


pr = export_model.predict(examples)

for i in range(len(examples)):
    if pr[i] < 0.5:
        ans = 0
    else:
        ans = 1
    print(examples[i], raw_train_ds.class_names[ans])
