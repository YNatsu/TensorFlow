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
from sklearn.metrics import r2_score
import keras.backend as K
from  sklearn.svm import SVR
sns.set()

housing = load_boston()

x_train, x_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=0, shuffle=True, test_size=0.25
)

print(housing.data.shape, housing.target.shape)
# (506, 13) (506,)

model = keras.models.Sequential(layers=[
    keras.layers.Dense(13, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b / e
    return f


model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae',r2]
)

callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-5)]

history = model.fit(
    x_train, y_train, epochs=64, validation_data=[x_test, y_test], callbacks=callbacks
)

y_pred = model.predict(x_train)
print(type(y_pred), type(y_train))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(r2_score(y_true=y_train, y_pred=y_pred))

svr = SVR()
svr.fit(x_train, y_train)
print(svr.score(x_train, y_train))
print(svr.score(x_test, y_test))

df = pd.DataFrame(history.history)
df.plot()
plt.show()
