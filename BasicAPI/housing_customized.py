# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/20 17:02
# @Title:  : 

import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

housing = load_boston()

x_train, x_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=0, shuffle=True, test_size=0.25
)

print(housing.data.shape, housing.target.shape)


# (506, 13) (506,)

class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomizedDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.layers.Activation(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_shape[1], self.units],
            initializer='uniform',
            trainable=True
        )

        self.bias = self.add_weight(
            name='bias',
            shape=[self.units, ],
            initializer='zero',
            trainable=True
        )
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        return self.activation(x @ self.kernel + self.bias)


def customized_mse(y, y_):
    return tf.reduce_mean(tf.square(y - y_))


custimized_softmax = keras.layers.Lambda(lambda x: tf.nn.softmax(x))

model = keras.models.Sequential(layers=[
    CustomizedDenseLayer(13, activation='relu', input_shape=[13]),
    CustomizedDenseLayer(1),
    custimized_softmax
])

model.summary()

model.compile(
    loss=customized_mse,
    optimizer='adam',
    metrics=['mean_squared_error']
)

callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-5)]

history = model.fit(
    x_train, y_train, epochs=64, validation_data=[x_test, y_test], callbacks=callbacks
)

df = pd.DataFrame(history.history)
df.plot()
plt.show()
