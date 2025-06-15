import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import itertools
import matplotlib.pyplot as plt



def f(x):
   return x*np.sin(8*x)-2


def pinball_loss(tau):
    def loss(y_true, y_pred):
        diff = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * diff, (tau - 1) * diff))
    return loss


class QuantileNeuralNetworkOptimized:
    def __init__(self, X, y, validation_split=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=42)
        self.validation_split = validation_split
        self.models = {}
        self.predictions_train = {}
        self.predictions_test = {}
        self.best_params = {}
        self.optimization_results = {}
    
    def create_model(self, quantile, hidden_layers=2, neurons_per_layer=64, 
                    learning_rate=0.01, dropout_rate=0.0, activation='relu'):

        model = keras.Sequential()
        
        # Capa de entrada
        model.add(layers.Dense(neurons_per_layer, activation=activation, 
                              input_shape=(self.X_train.shape[1],)))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
        
        # Capas ocultas
        for _ in range(hidden_layers - 1):
            model.add(layers.Dense(neurons_per_layer, activation=activation))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))
        
        # Capa de salida
        model.add(layers.Dense(1))
        
        # Compilación
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=pinball_loss(quantile)
        )
        
        return model

   
    def random_search_optimization(self, quantile, n_iter=50, param_distributions=None):
        """Optimización usando Random Search (más eficiente)"""
        print(f"Random Search para quantile {quantile} ({n_iter} iteraciones)")
        
        if param_distributions is None:
            param_distributions = {
                'hidden_layers': [1, 2, 3, 4],
                'neurons_per_layer': [16, 32, 64, 128, 256],
                'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4],
                'batch_size': [8, 16, 32, 64, 128],
                'epochs': [50, 100, 150, 200, 300]
            }
        
        best_score = float('inf')
        best_params = None
        results = []
        
        for i in range(n_iter):
            # Seleccionar parámetros aleatorios
            params = {}
            for key, values in param_distributions.items():
                params[key] = np.random.choice(values)
            
            # Validación usando validation_split
            model = self.create_model(
                quantile=quantile,
                hidden_layers=params['hidden_layers'],
                neurons_per_layer=params['neurons_per_layer'],
                learning_rate=params['learning_rate'],
                dropout_rate=params['dropout_rate']
            )

            print(f"Entrenando modelo con parámetros: {params}")
            callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

            history = model.fit(
                self.X_train, self.y_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                validation_split=self.validation_split,
                verbose=0,
                callbacks=[callback]
            )

            print (f"Finalizado entrenamiento para iteración {i+1}/{n_iter}")
            
            # Evaluar en conjunto de validación
            val_loss = min(history.history['val_loss'])
            
            results.append({
                'params': params.copy(),
                'score': val_loss
            })

            print(f"Iteración {i+1}/{n_iter}, Pérdida de validación: {val_loss:.4f}")
            
            if val_loss < best_score:
                best_score = val_loss
                best_params = params.copy()
            
            if (i + 1) % 10 == 0:
                print(f"Progreso: {i+1}/{n_iter}, Mejor score: {best_score:.4f}")
        
        self.best_params[quantile] = best_params
        self.optimization_results[quantile] = {
            'best_score': best_score,
            'best_params': best_params,
            'all_results': results
        }
        
        print(f"✅ Mejores parámetros para quantile {quantile}:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"Score: {best_score:.4f}")
        
        return best_params, best_score
    
    
    def fit(self, quantile: float):
        print(f"---------------------------------------Entrenando modelo para quantile: {quantile} -----------------------------------")
        
        # Optimización de hiperparámetros
        best_params, best_score = self.random_search_optimization(quantile)
        
        # Crear y entrenar el modelo final con los mejores parámetros
        model = self.create_model(
            quantile=quantile,
            hidden_layers=best_params['hidden_layers'],
            neurons_per_layer=best_params['neurons_per_layer'],
            learning_rate=best_params['learning_rate'],
            dropout_rate=best_params['dropout_rate']
        )

        callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        
        model.fit(
            self.X_train, self.y_train,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            validation_split=self.validation_split,
            verbose=0,
            callbacks=[callback]
        )
        
        self.models[quantile] = model
        print(f"Modelo para quantile {quantile} entrenado y guardado.")

    def predict(self, quantile: float):
        if quantile not in self.models:
            raise ValueError(f"Model for quantile {quantile} not trained yet")
        
        model = self.models[quantile]
        self.predictions_train[quantile] = model.predict(self.X_train, verbose=0)
        self.predictions_test[quantile] = model.predict(self.X_test, verbose=0)
        
        return self.predictions_test[quantile]
    

    def score_with_test(self, quantile: float):
        score = mean_pinball_loss(self.y_test, self.predictions_test[quantile], alpha=quantile)
        return score
    
    def score_with_train(self, quantile: float):
        score = mean_pinball_loss(self.y_train, self.predictions_train[quantile], alpha=quantile)
        return score
    
    def scores(self):
        print("Scores with test set:")
        for quantile in self.models.keys():
            score = self.score_with_test(quantile)
            print(f"Quantile {quantile}: {score}")
        
        print("Scores with train set:")
        for quantile in self.models.keys():
            score = self.score_with_train(quantile)
            print(f"Quantile {quantile}: {score}")
    

    def plot_all(self):
        xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
        y_lower = self.models[0.05].predict(xx)
        y_upper = self.models[0.95].predict(xx)
        y_med = self.models[0.5].predict(xx)

        fig = plt.figure(figsize=(10, 10))
        plt.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(8x)-2$")
        plt.plot(self.X_test, self.y_test, "b.", markersize=10, label="Test observations")
        plt.plot(xx, y_med, "r-", label="Predicted median")
        plt.plot(xx, y_upper, "k-")
        plt.plot(xx, y_lower, "k-")
        plt.fill_between(
            xx.ravel(), y_lower.ravel(), y_upper.ravel(), alpha=0.4, label="Predicted 90% interval"
        )
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        plt.ylim(-10, 25)
        plt.legend(loc="upper left")
        plt.show()
