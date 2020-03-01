# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/20 22:04
# @Title:  :

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
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

x_train_all = x_train_all.reshape([-1, 28, 28, 1])
x_test = x_test.reshape([-1, 28, 28, 1])

x_train_all = x_train_all.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

x_val, x_train = x_train_all[:5000], x_train_all[5000:]
y_val, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)

# (55000, 28, 28, 1) (55000,)
# (5000, 28, 28, 1) (5000,)
# (10000, 28, 28, 1) (10000,)


model = keras.models.Sequential(layers=[
    keras.layers.SeparableConv2D(filters=32, kernel_size=3, padding='same', activation='selu', input_shape=[28, 28, 1]),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding='same', activation='selu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='selu'),
    keras.layers.Dense(10, activation='softmax')
])

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 32)        320
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 14, 14, 64)        18496
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
# _________________________________________________________________
# flatten (Flatten)            (None, 3136)              0
# _________________________________________________________________
# dense (Dense)                (None, 128)               401536
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1290
# =================================================================
# Total params: 421,642
# Trainable params: 421,642
# Non-trainable params: 0
# _________________________________________________________________

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# separable_conv2d (SeparableC (None, 28, 28, 32)        73
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
# _________________________________________________________________
# separable_conv2d_1 (Separabl (None, 14, 14, 64)        2400
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
# _________________________________________________________________
# flatten (Flatten)            (None, 3136)              0
# _________________________________________________________________
# dense (Dense)                (None, 128)               401536
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1290
# =================================================================
# Total params: 405,299
# Trainable params: 405,299
# Non-trainable params: 0
# _________________________________________________________________

model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

logdir = os.path.join('fashion_mnist_callbacks')
if not os.path.exists(logdir): os.mkdir(logdir)

output_model_file = os.path.join(logdir, 'fashion_mnist_model.h5')

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]

history = model.fit(
    x_train, y_train, epochs=8, validation_data=[x_val, y_val], callbacks=callbacks
)

df = pd.DataFrame(history.history)
print(df)

#        loss  accuracy  val_loss  val_accuracy
# 0  0.415630  0.853527  0.290707        0.9000
# 1  0.271498  0.900909  0.278700        0.8976
# 2  0.224923  0.917127  0.247125        0.9090
# 3  0.192133  0.929509  0.218881        0.9190
# 4  0.166497  0.937018  0.246038        0.9108
# 5  0.142661  0.946891  0.218921        0.9202
# 6  0.119235  0.954582  0.228378        0.9242
# 7  0.099689  0.962200  0.247079        0.9146

df.plot()
plt.show()

eva = model.evaluate(x_test, y_test)
print(eva)
