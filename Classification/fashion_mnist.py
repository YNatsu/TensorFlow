# encoding: utf-8
# @Author  :   YNatsu 
# @Time    :   2020/2/19 17:11
# @Title:  :   fashion mnist classification

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn

seaborn.set()

print(tf.__version__)
# 2.0.0

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

x_train_all = x_train_all.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

x_val, x_train = x_train_all[:5000], x_train_all[5000:]
y_val, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)

# (55000, 28, 28) (55000,)
# (5000, 28, 28) (5000,)
# (10000, 28, 28) (10000,)

# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))


model = tf.keras.models.Sequential(layers=[
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation='relu'),
    # keras.layers.BatchNormalization(),
    # keras.layers.AlphaDropout(0.5),
    tf.keras.layers.Dense(300, activation='relu'),
    # keras.layers.BatchNormalization(),
    # keras.layers.AlphaDropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# flatten (Flatten)            (None, 784)               0
# _________________________________________________________________
# dense (Dense)                (None, 300)               235500
# _________________________________________________________________
# dense_1 (Dense)              (None, 300)               90300
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                3010
# =================================================================
# Total params: 328,810
# Trainable params: 328,810
# Non-trainable params: 0
# _________________________________________________________________

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

logdir = os.path.join('fashion_mnist_callbacks')
if not os.path.exists(logdir): os.mkdir(logdir)

output_model_file = os.path.join(logdir, 'fashion_mnist_model.h5')

callbacks = [
    tf.keras.callbacks.TensorBoard(logdir),
    tf.keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]

history = model.fit(
    x_train, y_train, epochs=8, validation_data=[x_val, y_val], callbacks=callbacks
)

df = pd.DataFrame(history.history)
print(df)

#        loss  accuracy  val_loss  val_accuracy
# 0  2.653283  0.766436  0.611606        0.8074
# 1  0.524818  0.823855  0.547937        0.8264
# 2  0.473799  0.836364  0.435519        0.8510
# 3  0.444093  0.842927  0.475706        0.8366

df.plot()
plt.show()

eva = model.evaluate(x_test, y_test)
print(eva)
