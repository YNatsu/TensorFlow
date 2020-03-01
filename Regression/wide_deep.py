# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/20 10:25
# @Title:  :    wide and deep model


from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *
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

# input = keras.layers.Input([13, ])
# h1 = keras.layers.Dense(30, activation='relu')(input)
# h2 = keras.layers.Dense(30, activation='relu')(h1)
#
# concat = keras.layers.concatenate([input, h2])
# output = keras.layers.Dense(1)(concat)
#
# model = keras.models.Model(inputs=[input], outputs=output)


class WideDeepModel(Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        self.h1 = Dense(30, activation='relu')
        self.h2 = Dense(30, activation='relu')
        self.o = Dense(1)

    def call(self, input):
        h1 = self.h1(input)
        h2 = self.h2(h1)
        concat = concatenate([input, h2])
        o = self.o(concat)
        return o


model = WideDeepModel()
model.build(input_shape=(None, 13))
model.summary()

# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            [(None, 13)]         0
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 30)           420         input_1[0][0]
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 30)           930         dense[0][0]
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 43)           0           input_1[0][0]
#                                                                  dense_1[0][0]
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 1)            44          concatenate[0][0]
# ==================================================================================================
# Total params: 1,394
# Trainable params: 1,394
# Non-trainable params: 0
# __________________________________________________________________________________________________

logdir = os.path.join('wide_deep_callbacks')
if not os.path.exists(logdir): os.mkdir(logdir)

output_model_file = os.path.join(logdir, 'wide_deep_model.h5')

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
)

callbacks = [
    TensorBoard(logdir),
    ModelCheckpoint(output_model_file, save_best_only=True),
    EarlyStopping(patience=5, min_delta=1e-3)
]

history = model.fit(
    x_train, y_train, epochs=64, validation_data=[x_test, y_test], callbacks=callbacks
)

df = pd.DataFrame(history.history)
df.plot()
plt.show()
