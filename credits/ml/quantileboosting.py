from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV 
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np


class QBR:
    def __init__(self, X, y ):
        self.X_train, self.X_test,self. y_train, self.y_test = train_test_split(X, y, random_state=0)
        self.models = {}
        self.predictions_train = {}
        self.predictions_test = {}

    # Scorer para la función de pérdida pinball
    def neg_mean_pinball_loss(self, alpha):
        return make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False, )
    
    def fit(self, alpha: float):
        learning_rate = np.logspace(-3, -0.5, 20)  # De 0.001 a ~0.316
        max_depth = np.arange(2, 12, 1)
        min_samples_leaf = np.arange(1, 31, 2)
        min_samples_split = np.arange(5, 101, 5)

        param_grid = dict(
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
        )

        neg_mean_pinball_loss_scorer= self.neg_mean_pinball_loss(alpha)

        gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=0)

        #busqueda de hiperparámetros aleatoria
        search_p = HalvingRandomSearchCV(
            gbr,
            param_grid,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=1,
            random_state=0,
            verbose = 0,
            cv=5
        ).fit(self.X_train, self.y_train)

        self.models[alpha] = search_p
        

    def predict(self, alpha:float, X):
        prediction = self.models[alpha].predict(X)
        return prediction
    
    def score(self, alpha: float, X, y):
        prediction = self.predict(alpha, X)
        score = mean_pinball_loss(y, prediction, alpha=alpha)
        return score

