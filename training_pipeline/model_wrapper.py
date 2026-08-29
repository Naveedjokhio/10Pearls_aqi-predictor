import numpy as np


class KerasMLPRegressor:

    def __init__(self, input_dim, epochs=80, batch_size=8, random_state=42):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._weights = None
        self._mean = None
        self._std = None
        self._y_mean = None
        self._y_std = None
        self._model = None  

    def _build(self):
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)
        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X, y):
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")

        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        X_scaled = (X - self._mean) / self._std

        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 0 else 1.0
        y_scaled = (y - self._y_mean) / self._y_std

        model = self._build()
        model.fit(X_scaled, y_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self._weights = model.get_weights()
        self._model = model
        return self

    def predict(self, X):
        X = np.asarray(X, dtype="float32")
        X_scaled = (X - self._mean) / self._std
        if self._model is None:
            self._model = self._build()
            self._model.set_weights(self._weights)
        preds_scaled = self._model.predict(X_scaled, verbose=0).flatten()
        return preds_scaled * self._y_std + self._y_mean

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None  
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None