from quantileregression import QR
from expectileregression import ER
import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd

def f(x):
   return x*np.sin(8*x)-2


if __name__ == "__main__":
    rng = np.random.RandomState(56)
    X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
    y = f(X).ravel() + (rng.chisquare(df=3, size=X.shape[0]) - 3) * 1.5
    #data = fetch_california_housing()
    
    #X = pd.DataFrame(data.data, columns=data.feature_names)
    #y = data.target

    credit =ER(X, y)
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
    


    


