# encoding: utf-8 
# @Author  :   YNatsu 
# @Time    :   2020/2/20 12:04
# @Title:  :    Hyper parameter search

from tensorflow import keras
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns

sns.set()

housing = load_boston()

x_train, x_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=0, shuffle=True, test_size=0.25
)

print(housing.data.shape, housing.target.shape)


# (506, 13) (506,)


def build_model(nums=30, learning_rate=1e-3):
    model = keras.models.Sequential(layers=[
        keras.layers.Dense(nums, activation='relu'),
        keras.layers.Dense(1)
    ])

    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )
    return model


paras = {
    'nums': [10, 20, 30],
    'learning_rate': reciprocal(1e-4, 1e-2)
}

sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)

random_search = RandomizedSearchCV(
    sklearn_model, param_distributions=paras,
    n_iter=10, n_jobs=1
)

callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-5)]

random_search.fit(
    x_train, y_train, epochs=64, validation_data=[x_test, y_test], callbacks=callbacks
)

print(random_search.best_params_)
# {'learning_rate': 0.00580076999074848, 'nums': 30}
print(random_search.best_score_)
print(random_search.best_estimator_)
