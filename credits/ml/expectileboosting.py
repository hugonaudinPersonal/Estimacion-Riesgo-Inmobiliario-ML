import numpy as np
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import make_scorer
from lightgbm.sklearn import LGBMRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

## Función de pérdida expectile -> gradiente y hessiano
def expectile_loss(tau):
    def loss(y_true, y_pred):
        residual = y_true - y_pred
        grad = -2 * np.where(residual < 0, 1 - tau, tau) * residual
        hess = 2 * np.where(residual < 0, 1 - tau, tau)
        return grad, hess
    return loss

##Clase del expectile boosting con LightGBM
class ExpectileLGBM(LGBMRegressor):
    def __init__(self, tau=0.5, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
    
    def fit(self, X, y, **kwargs):
        self.set_params(objective=expectile_loss(self.tau))
        return super().fit(X, y, **kwargs)



class EBR:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0)
        self.models = {}
        self.predictions_train = {}
        self.predictions_test = {}
        self.tau = None

    #Scorer para la función de pérdida expectile
    def neg_mean_expectile_loss_scorer(self, expectile):
        def score_fn(y_true, y_pred):
            errors = y_true - y_pred
            return -np.mean(np.where(errors < 0, 1 - expectile, expectile) * errors ** 2)
        return make_scorer(score_fn, greater_is_better=False)
    

    def fit(self, expectile: float):

        print (f"Fitting Expectile Boosting model for expectile {expectile}...")
        self.tau = expectile

        param_grid = {
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "max_depth": Integer(2, 12),
            "min_data_in_leaf": Integer(5, 50),
            "num_leaves": Integer(15, 150),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "reg_alpha": Real(1e-4, 1.0, prior="log-uniform"),
            "reg_lambda": Real(1e-4, 1.0, prior="log-uniform"),
            "min_split_gain": Real(0, 0.3),
            "max_bin": Integer(63, 255)
        }


        neg_mean_expectile_loss_scorer = self.neg_mean_expectile_loss_scorer(expectile)


        base_model = ExpectileLGBM(
            tau=expectile,
            random_state=0,
            verbose=-1,
            n_estimators=300  # En lugar de early stopping
        )

        # Busqueda bayesiana de hiperparámetros
        search = BayesSearchCV(
            estimator=base_model,
            search_spaces=param_grid,
            scoring=neg_mean_expectile_loss_scorer,
            n_iter=40,
            cv=3,
            n_jobs=-1,
            random_state=0,
            verbose=0
        )

        search.fit(self.X_train, self.y_train)

        self.models[expectile] = search.best_estimator_

        print(f"Best score for expectile {expectile}: {search.best_score_}")

    def predict(self, expectile, X): 
        prediction = self.models[expectile].predict(X)
        return prediction

    def score(self, alpha: float, X, y):
        prediction = self.predict(alpha, X)
        errors = y - prediction
        return np.mean(np.where(errors < 0, 1 - alpha, alpha) * errors ** 2)


