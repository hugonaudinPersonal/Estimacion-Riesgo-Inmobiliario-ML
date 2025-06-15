from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV 
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x*np.sin(8*x)-2


class QBR:
    def __init__(self, X, y ):
        self.X_train, self.X_test,self. y_train, self.y_test = train_test_split(X, y, random_state=0)
        self.models = {}
        self.predictions_train = {}
        self.predictions_test = {}

        
    def neg_mean_pinball_loss(self, alpha):
        return make_scorer(mean_pinball_loss, alpha=alpha,greater_is_better=False, )
    
    def fit(self, alpha: float):
        learning_rate = np.arange(0.01, 0.31, 0.01)
        max_depth = np.arange(1, 11, 1)
        min_samples_leaf = np.arange(1, 21, 1)
        min_samples_split = np.arange(5, 51, 1)

        param_grid = dict(
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split
        )

        neg_mean_pinball_loss_scorer= self.neg_mean_pinball_loss(alpha)

        gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=0)
        print(f"---------------------------------------Training model for alpha: {alpha} ---------------------------------------")
        search_p = HalvingRandomSearchCV(
            gbr,
            param_grid,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=2,
            random_state=0,
            verbose = 2
        ).fit(self.X_train, self.y_train)

        self.models[alpha] = search_p
        
    
    def predict(self, alpha: float):
        prediction = self.models[alpha].predict(self.X_test)
        self.predictions_test[alpha] = prediction
        prediction = self.models[alpha].predict(self.X_train)
        self.predictions_train[alpha] = prediction
    
    def score_with_test(self, alpha: float):
        score = mean_pinball_loss(self.y_test, self.predictions_test[alpha], alpha=alpha)
        return score
    
    def score_with_train(self, alpha: float):
        score = mean_pinball_loss(self.y_train, self.predictions_train[alpha], alpha=alpha)
        return score
    
    def scores(self):
        for alpha in self.models.keys():
            score = self.score_with_train(alpha)
            print(f"Pinball loss for alpha with train {alpha}: {score}")
            score = self.score_with_test(alpha)
            print(f"Pinball loss for alpha with test {alpha}: {score}")
            
##Métodos utilizados unicamente para datasets de prueba
    def plot(self, alpha):
       xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
       plt.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(8x)-2$")
       plt.plot(self.X_test, self.y_test, "b.", markersize=10, label="Test observations")
       plt.plot(xx, self.models[alpha].predict(xx), "k-")
       plt.xlabel("$x$")
       plt.ylabel("$f(x)$")
       plt.ylim(-10, 25)
       plt.legend(loc="upper left")
       plt.show()
#       
#
#
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
#
#
        
