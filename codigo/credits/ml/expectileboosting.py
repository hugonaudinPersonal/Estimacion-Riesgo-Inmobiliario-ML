import lightgbm as lgb
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from lightgbm.sklearn import LGBMRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def expectile_loss(tau):
    def loss(y_true, y_pred):
        residual = y_true - y_pred
        grad = -2 * np.where(residual < 0, 1 - tau, tau) * residual
        hess = 2 * np.where(residual < 0, 1 - tau, tau)
        return grad, hess
    return loss

class ExpectileLGBM(LGBMRegressor):
    def __init__(self, tau=0.5, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
    
    def fit(self, X, y, **kwargs):
        self.set_params(objective=expectile_loss(self.tau))
        return super().fit(X, y, **kwargs)




def f(x):
    return x * np.sin(8 * x) - 2
class EBR:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0)
        self.models = {}
        self.predictions_train = {}
        self.predictions_test = {}

    def neg_mean_expectile_loss_scorer(self, expectile):
        def score_fn(y_true, y_pred):
            errors = y_true - y_pred
            return -np.mean(np.where(errors < 0, 1 - expectile, expectile) * errors ** 2)
        return make_scorer(score_fn, greater_is_better=False)
    

    def fit(self, expectile: float):
        learning_rate = np.arange(0.01, 0.31, 0.01)
        max_depth = np.arange(1, 11, 1)
        min_data_in_leaf = np.arange(1, 21, 1)

        param_grid = dict(
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_data_in_leaf=min_data_in_leaf,
        )

        neg_mean_expectile_loss_scorer = self.neg_mean_expectile_loss_scorer(expectile)

        ebr  = ExpectileLGBM(tau=expectile, random_state=0, verbose=-1)

        print(f"---------------------------------------Training model for expectile: {expectile} ---------------------------------------")
        
        search_p = HalvingRandomSearchCV(
            ebr,
            param_grid,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_expectile_loss_scorer,
            n_jobs=2,
            random_state=0,
            verbose=2
        ).fit(self.X_train, self.y_train)

        self.models[expectile] = search_p

    def predict(self, expectile: float):
        prediction = self.models[expectile].predict(self.X_test)
        self.predictions_test[expectile] = prediction
        prediction = self.models[expectile].predict(self.X_train)
        self.predictions_train[expectile] = prediction

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
    def plot(self, tau):
        xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
        plt.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(8x)-2$")
        plt.plot(self.X_test, self.y_test, "b.", markersize=10, label="Test observations")
        plt.plot(xx, self.models[tau].predict(xx), "k-")
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        plt.ylim(-10, 25)
        plt.legend(loc="upper left")
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
#


