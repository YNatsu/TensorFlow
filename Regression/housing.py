# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/19 18:34
# @Title:  :   housing regression

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

model = keras.models.Sequential(layers=[
    keras.layers.Dense(13, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
)
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-5)]

history = model.fit(
    x_train, y_train, epochs=64, validation_data=[x_test, y_test], callbacks=callbacks
)

df = pd.DataFrame(history.history)
df.plot()
plt.show()
