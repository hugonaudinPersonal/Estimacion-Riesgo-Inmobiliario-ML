from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator




### Implementacion de Expectile Regression Lineal
class ExpectileRegression(BaseEstimator):

    #Metodo necesario    
    def __init__(self, expectile: float =.5, **kwargs):
        assert 0 < expectile < 1, \
            f"expectile parameter must be stricly 0 < e < 1, but it equals: {expectile}"
        self.expectile = expectile
        super().__init__(**kwargs)

    #Metodo necesario
    def get_params(self, deep=True):
        return {'expectile': self.expectile}
    
    #Metodo necesario
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid parameter {key} for estimator {self.__class__.__name__}')
        return self
        
    #Entrenamiento del modelo    
    def fit(self, X, y):
        
        if type(X) is pd.DataFrame:
            X = X.values
        
        if type(y) in [pd.DataFrame, pd.Series]:
            y = y.values
        
        # Adding column of ones
        n = X.shape[0]

        #Se pone  una columna de unos para el intercepto
        X = np.hstack([X, np.ones(shape=(n, 1))])
        
        self.beta = np.random.rand(X.shape[1])
        
        #Perdida
        def expectile_loss(beta, *args):
            y_hat = X @ beta
            errors = y.reshape(-1, 1) - y_hat.reshape(-1, 1)
            return (
                    np.where(
                        errors < 0, 1 - self.expectile , self.expectile
                    ) * errors ** 2
                   ).mean()
        
        # Gradiente de la perdida
        def expectile_grad(beta, *args):
            y_hat = X @ beta
            errors = y.reshape(-1, 1) - y_hat.reshape(-1, 1)
            return (
                    -2 * X.T @ (
                        np.where(
                            errors < 0, 1 - self.expectile , self.expectile
                        ) * errors
                      )
                    ) / X.shape[0]
        
        # Minimizacion
        res = minimize(
            fun    = expectile_loss,
            x0     = self.beta,
            args   = None,
            method = 'SLSQP' ,
            jac    = expectile_grad
        )
        
        self.beta = res.x
        
        return self
    
    #Prediccion del modelo
    def predict(self, X):
        n = X.shape[0]
        X_ = np.hstack([X, np.ones(shape=(n, 1))])
        return X_ @ self.beta
    


class ER:
    def __init__(self, X, y ):
        self.X_train, self.X_test,self. y_train, self.y_test = train_test_split(X, y, random_state=0)
        self.models = {}
        self.predictions_train = {}
        self.predictions_test = {}

    
    def neg_mean_expectile_loss_scorer(self, expectile):
        def score_fn(y_true, y_pred):
            errors = y_true - y_pred
            return -np.mean(np.where(errors < 0, 1 - expectile, expectile) * errors ** 2)
        
        return make_scorer(score_fn, greater_is_better=False)

    def fit(self, expectile: float):
        print(f"---------------------------------------Training model for expectile: {expectile} -----------------------------------")

        model = ExpectileRegression(expectile=expectile)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        #Validacion cruzada
        scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=kf,
            scoring=self.neg_mean_expectile_loss_scorer(expectile),
            n_jobs=1
        )
        score = scores.mean()
        print(f"Score for expectile {expectile}: {score}")
        model.fit(self.X_train, self.y_train)

        self.models[expectile] = model

    def predict(self, expectile, X):
        prediction = self.models[expectile].predict(X)
        return prediction

    def score(self, alpha: float, X, y):
        prediction = self.predict(alpha, X)
        errors = y - prediction
        return np.mean(np.where(errors < 0, 1 - alpha, alpha) * errors ** 2)
    
