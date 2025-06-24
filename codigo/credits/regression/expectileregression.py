from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator


def f(x):
   return x*np.sin(8*x)-2


class ExpectileRegression(BaseEstimator):
    """Expectile regression with linear model"""
    
    def __init__(self, expectile: float =.5, **kwargs):
        assert 0 < expectile < 1, \
            f"expectile parameter must be stricly 0 < e < 1, but it equals: {expectile}"
        self.expectile = expectile
        super().__init__(**kwargs)


    def get_params(self, deep=True):
        return {'expectile': self.expectile}
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid parameter {key} for estimator {self.__class__.__name__}')
        return self
        
    def fit(self, X, y):
        
        if type(X) is pd.DataFrame:
            X = X.values
        
        if type(y) in [pd.DataFrame, pd.Series]:
            y = y.values
        
        # Adding column of ones
        n = X.shape[0]
        X = np.hstack([X, np.ones(shape=(n, 1))])
        
        self.beta = np.random.rand(X.shape[1])
        
        def expectile_loss(beta, *args):
            y_hat = X @ beta
            errors = y.reshape(-1, 1) - y_hat.reshape(-1, 1)
            return (
                    np.where(
                        errors < 0, 1 - self.expectile , self.expectile
                    ) * errors ** 2
                   ).mean()
        
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
        
        # Optimization
        res = minimize(
            fun    = expectile_loss,
            x0     = self.beta,
            args   = None,
            method = 'SLSQP' ,
            jac    = expectile_grad
        )
        
        self.beta = res.x
        
        return self
    
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
        scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=kf,
            scoring=self.neg_mean_expectile_loss_scorer(expectile),
            n_jobs=-1
        )
        score = scores.mean()
        print(f"Score for expectile {expectile}: {score}")
        model.fit(self.X_train, self.y_train)

        self.models[expectile] = model
    
    def predict(self, expectile: float):
        prediction = self.models[expectile].predict(self.X_test)
        self.predictions_test[expectile] = prediction
        prediction = self.models[expectile].predict(self.X_train)
        self.predictions_train[expectile] = prediction

    def predict(self, expectile, X):
        prediction = self.models[expectile].predict(X)
        return prediction

    def score_with_test(self, expectile: float):
        errors = self.y_test - self.predictions_test[expectile]
        return np.mean(np.where(errors < 0, 1 - expectile, expectile) * errors ** 2)
    
    def score_with_train(self, expectile: float):
        errors = self.y_train - self.predictions_train[expectile]
        return np.mean(np.where(errors < 0, 1 - expectile, expectile) * errors ** 2)
    
    def scores(self):
        print("Scores with test set:")
        for expectile in self.models.keys():
            score = self.score_with_test(expectile)
            print(f"Expectile {expectile}: {score}")
        
        print("Scores with train set:")
        for expectile in self.models.keys():
            score = self.score_with_train(expectile)
            print(f"Expectile {expectile}: {score}")
            
##Métodos utilizados unicamente para datasets de prueba
    def plot(self, X, y):
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Data')
        for expectile in self.models.keys():
            predictions = self.models[expectile].predict(X)
            plt.plot(X, predictions, label=f'Expectile {expectile}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Expectile Regression')
        plt.legend()
        plt.show()


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
            xx.ravel(), y_lower, y_upper, alpha=0.4, label="Predicted 90% interval"
        )
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        plt.ylim(-10, 25)
        plt.legend(loc="upper left")
        plt.show()
 