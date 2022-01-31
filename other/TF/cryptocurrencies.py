import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
import time
import tensorflow as tf


'''
READING DATA, MERGING DATA, CREATING FUTURE, CREATING LABELS FOR DATA
'''


SEQ_LEN = 60                    # LAST 60 MINUTES
FUTURE_PRED_DIST = 3            # PREDICT 3 MINUTES FROM NOW
RATIO_TO_PREDICT = 'LTC-USD'    # WHICH CURRENCY WE PREDICT


def classify(current, future):
    if float(future) > float(current):
        return 1    # a good event, means we should buy
    else:
        return 0    # not good event, we shouldn't buy


main_df = pd.DataFrame()


for ratio in os.listdir('crypto_data'):
    ratio_name = ratio[:-4]
    df = pd.read_csv(f'crypto_data/{ratio}', names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.rename(columns={'close': f'{ratio_name}_close', 'volume': f'{ratio_name}_volume'}, inplace=True)
    #print(df.head())
    df.set_index('time', inplace=True)
    df = df[[f'{ratio_name}_close', f'{ratio_name}_volume']]
    #print(df.head())

    if main_df.empty:
        main_df = df
    else:
        main_df = main_df.join(df)

    #print(main_df.head())


main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PRED_DIST)
print(main_df.head())
print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future']].head(10))

#main_df['target'] = [classify(c, f) for c, f in list(zip(main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))]
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))
print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future', 'target']].head(10))


'''
SEPARATING TRAINING AND VALIDATION DATA, CREATING SEQUENTIAL DATA FROM DATA FRAMES
'''


def trainable_data(df):
    df.drop('future', 1)
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for value in df.values:
        prev_days.append([n for n in value[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), value[-1]])

    np.random.shuffle(sequential_data)
    return sequential_data


def balance_data(data):
    buys = []
    sells = []
    print(len(data))
    for d in data:
        if d[-1] == 0:
            sells.append(d)
        else:
            buys.append(d)
    print(len(buys), len(sells))
    low = min(len(buys), len(sells))
    buys = buys[:low]
    sells = sells[:low]
    newdata = buys + sells
    np.random.shuffle(newdata)
    return newdata


def split_labels(data):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    return np.array(x), np.array(y)


timestamps = sorted(main_df.index.values)
VALIDATION_FRACTION = 0.05
smallest_valid_timestamp = timestamps[-int(VALIDATION_FRACTION * len(timestamps))]
print(smallest_valid_timestamp)

validation_df = main_df[(main_df.index >= smallest_valid_timestamp)]
training_df = main_df[(main_df.index < smallest_valid_timestamp)]

print(training_df.head())
print(validation_df.head())

train_data = trainable_data(training_df)
test_data = trainable_data(validation_df)
train_data = balance_data(train_data)

x_train, y_train = split_labels(train_data)
x_test, y_test = split_labels(test_data)

print(x_train[0], y_train[0])
print(x_test[0], y_test[0])


'''

'''
EPOCHS = 10
BATCH_SIZE = 64
NAME = f'{SEQ_LEN}-seq-{FUTURE_PRED_DIST}-dist-{RATIO_TO_PREDICT}-ratio-{int(time.time())}'

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(units=64, input_shape=x_train.shape[1:], return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.LSTM(units=64, return_sequences=False))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

tfboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{NAME}')
filepath = 'RNN_checkpoint'
checkpoint = tf.keras.callbacks.ModelCheckpoint(f'models/{filepath}.model', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit(
    x=x_train, y=y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[tfboard, checkpoint]
)

model.save('')