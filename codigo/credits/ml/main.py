from quantileboosting import QBR
from expectileboosting import EBR
from neuralnetwork import QuantileNeuralNetworkOptimized
import numpy as np
from sklearn.datasets import fetch_california_housing

##
def f(x):
    return x*np.sin(8*x)-2
#

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
    sigma = 0.5 + X.ravel() / 10
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    y = f(X).ravel()+noise
    #sigma = 0.5 + X.ravel() / 10
    #noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    #y = y + noise
    data = fetch_california_housing()
    X = data.data
    y = data.target

    credit = QBR(X, y)
    #Entrenamiento del modelo
    credit.fit(0.95)
    credit.predict(0.95)


    #credit.plot(0.95)

    credit.fit(0.5)
    credit.predict(0.5)
    #credit.plot(0.5)

    credit.fit(0.05)
    credit.predict(0.05)
    #credit.plot(0.05)
    credit.scores()

    credit.plot_all()

    


    


