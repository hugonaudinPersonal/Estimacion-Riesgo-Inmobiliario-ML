from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_pinball_loss
import numpy as np
import matplotlib.pyplot as plt
import sys

def f(x):
   return x*np.sin(8*x)-2



class QR:
    def __init__(self, X, y ):
        self.X_train, self.X_test,self. y_train, self.y_test = train_test_split(X, y, random_state=0)
        self.models = {}
        self.predictions_train = {}
        self.predictions_test = {}

        
    def neg_mean_pinball_loss_scorer(self, alpha):
        return make_scorer(mean_pinball_loss, alpha=alpha,greater_is_better=False, )
    
    def fit(self, alpha: float):
        print(f"---------------------------------------Training model for alpha: {alpha} -----------------------------------")


        
        model = QuantileRegressor(quantile=alpha, alpha = 0)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

    
        scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=kf,
            scoring=self.neg_mean_pinball_loss_scorer(alpha), 
            n_jobs=-1 
        )

        score = scores.mean()

        

        print(f"Score for alpha {alpha}: {score}")
        model.fit(self.X_train, self.y_train)

        self.models[alpha] = model

        


    
    def predict(self, alpha: float):
        prediction = self.models[alpha].predict(self.X_test)
        self.predictions_test[alpha] = prediction
        prediction = self.models[alpha].predict(self.X_train)
        self.predictions_train[alpha] = prediction

    def predict(self, alpha, X):
        prediction = self.models[alpha].predict(X)
        return prediction
    
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
        fig = plt.figure(figsize=(10, 10))
        plt.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(8x)-2$")
        plt.plot(self.X_test, self.y_test, "b.", markersize=10, label="Test observations")
        plt.plot(xx, self.models[alpha].predict(xx), "k-")
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
 
 
        
