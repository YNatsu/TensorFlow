# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/20 11:10
# @Title:  :    multi input

import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set()

housing = load_boston()

x_train, x_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=0, shuffle=True, test_size=0.25
)

print(housing.data.shape, housing.target.shape)

# (506, 13) (506,)

input_wide = tf.keras.layers.Input(shape=[8])
input_deep = tf.keras.layers.Input(shape=[5])

h1 = tf.keras.layers.Dense(30, activation='relu')(input_deep)
h2 = tf.keras.layers.Dense(30, activation='relu')(h1)

concat = tf.keras.layers.concatenate([input_wide, h2])
output = tf.keras.layers.Dense(1)(concat)

model = tf.keras.models.Model(
    inputs=[input_wide, input_deep], outputs=[output]
)

model.summary()

logdir = os.path.join('wide_deep_multi_input_callbacks')
if not os.path.exists(logdir): os.mkdir(logdir)
output_model_file = os.path.join(logdir, 'wide_deep_multi_input_model.h5')

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
)

callbacks = [
    tf.keras.callbacks.TensorBoard(logdir),
    tf.keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]

x_train_wide = x_train[:, :8]
x_train_deep = x_train[:, 8:]

x_test_wide = x_test[:, :8]
x_test_deep = x_test[:, 8:]

print(x_train_wide.shape, x_train_deep.shape, y_train.shape)
print(x_test_wide.shape, x_test_deep.shape, y_test.shape)

history = model.fit(
    [x_train_wide, x_train_deep], y_train, epochs=64,
    validation_data=[[x_test_wide, x_test_deep], y_test],
    callbacks=callbacks
)

df = pd.DataFrame(history.history)
df.plot()
plt.show()
