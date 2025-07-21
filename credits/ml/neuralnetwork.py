import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss
from skopt import BayesSearchCV
from skopt.space import Real,Integer
from scikeras.wrappers import KerasRegressor
import gc



@tf.keras.utils.register_keras_serializable()
class PinballLoss(tf.keras.losses.Loss):
    def __init__(self, tau, name="pinball_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tau = tau

    def call(self, y_true, y_pred):
        diff = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.tau * diff, (self.tau - 1) * diff))

    def get_config(self):
        config = super().get_config()
        config.update({"tau": self.tau})
        return config


###Creacion el modelo segun hiperparametros
def create_quantile_model(quantile, input_shape, hidden_layers=2, neurons_per_layer=64,
                          learning_rate=0.01, dropout_rate=0.0, activation='relu', l2_penalty=1e-4):
    model = keras.Sequential()
    model.add(layers.Dense(neurons_per_layer,
                           input_shape=input_shape,
                           kernel_regularizer=regularizers.l2(l2_penalty)))
    model.add(layers.BatchNormalization()) 
    model.add(layers.Activation(activation))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    for _ in range(hidden_layers - 1):
        model.add(layers.Dense(neurons_per_layer, activation=activation, kernel_regularizer = regularizers.l2(l2_penalty)))
        model.add(layers.BatchNormalization())  
        model.add(layers.Activation(activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=PinballLoss(quantile)
    )
    return model


class QuantileNeuralNetwork:
    def __init__(self, X, y, validation_split=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=42)
        self.validation_split = validation_split
        self.quantiles = []
        self.predictions_train = {}
        self.predictions_test = {}
        self.best_params = {}
        self.optimization_results = {}

    def bayesian_search_optimization(self, quantile, n_iter=40, param_space=None):
        print(f"Bayesian Search para quantile {quantile} ({n_iter} iteraciones)")
        if param_space is None:
            param_space = {
                'model__hidden_layers': Integer(1, 7),
                'model__neurons_per_layer': Integer(16, 512),
                'model__learning_rate': Real(1e-4, 1e-1, prior='log-uniform'),
                'model__dropout_rate': Real(0.0, 0.6),
                'batch_size': Integer(8, 128),
                'epochs': Integer(50, 300)
            }
        
        reg = KerasRegressor(
            model=create_quantile_model,
            quantile=quantile,
            input_shape=(self.X_train.shape[1],)
        )

        opt = BayesSearchCV(
            reg,
            param_space,
            n_iter=n_iter,
            cv=3,
            n_jobs=-1,
            verbose=0,
            random_state=42
        )

        opt.fit(self.X_train, self.y_train)

        best_params = opt.best_params_
        best_score = opt.best_score_
        
        best_params = {k.replace('model__', ''): v for k, v in best_params.items()}

        self.best_params[quantile] = best_params
        self.optimization_results[quantile] = {
            'best_score': best_score,
            'best_params': best_params,
            'all_results': opt.cv_results_
        }
        
        print(f"Mejores parámetros para quantile {quantile}:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"Score: {best_score:.4f}")
        
        return best_params, best_score

    def fit(self, quantile: float):
        print(f"---------------------------------------Entrenando modelo para quantile: {quantile} -----------------------------------")
        
        #Busqueda de hiperparametros optimos
        best_params, best_score = self.bayesian_search_optimization(quantile)
        
        model = create_quantile_model(
            quantile=quantile,
            input_shape=(self.X_train.shape[1],),
            hidden_layers=best_params['hidden_layers'],
            neurons_per_layer=best_params['neurons_per_layer'],
            learning_rate=best_params['learning_rate'],
            dropout_rate=best_params['dropout_rate']
        )

        early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-4,
            verbose=0
        )
                
        model.fit(
            self.X_train, self.y_train,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            validation_split=self.validation_split,
            verbose=0,
            callbacks=[early_stopping, reduce_lr]
        )

        self.quantiles.append(quantile)
        model.save(f"ml/models/quantile_model_{quantile}.keras")

        del model
        model = None
        K.clear_session()
        gc.collect()
        print(f"Modelo para quantile {quantile} entrenado y guardado.")

    def predict(self, alpha, X):        
        print(f"Prediciendo quantile {alpha}...")
        model = keras.models.load_model(f"ml/models/quantile_model_{alpha}.keras", custom_objects={'PinballLoss': PinballLoss(alpha)})
        prediction = model.predict(X, verbose=0)
        return prediction

    def score(self, alpha: float, X, y):
        prediction = self.predict(alpha, X)
        score = mean_pinball_loss(y, prediction, alpha=alpha)
        return score

