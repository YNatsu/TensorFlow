# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/22 20:19
# @Title:  : 

import tensorflow as tf
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import pandas as pd

imdb = tf.keras.datasets.imdb

vocal_size = 10000
index_from = 3

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=vocal_size, index_from=index_from
)

print(train_data[0])
print(train_labels[0])

# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670,
# 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50,
# 16,6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515,
# 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38,
# 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117,
# 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15,
# 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28,
# 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]

# 1

max_length = 500

train_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_data,
    maxlen=max_length,
    padding='post'
)

test_data = tf.keras.preprocessing.sequence.pad_sequences(
    test_data,
    maxlen=max_length,
    padding='post'
)
print(len(train_data[0]))
# 500


embedding_dim = 16
batch_size = 128

model = tf.keras.models.Sequential(layers=[
    Embedding(vocal_size, embedding_dim, input_length=max_length),
    Bidirectional(SimpleRNN(units=32, return_sequences=False)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])



model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['accuracy']
)

history = model.fit(
    train_data, train_labels, epochs=8,
    batch_size=batch_size, validation_split=0.2
)

pd.DataFrame(history.history).plot()
plt.show()

model.evaluate(test_data, test_labels, batch_size=batch_size)
