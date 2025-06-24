from ml.expectileboosting import EBR
from ml.quantileboosting import QBR
from ml.neuralnetwork import QuantileNeuralNetworkOptimized
from regression.quantileregression import QR 
from regression.expectileregression import ER 
import numpy as np 
import matplotlib.pyplot as plt 

def f(x):
    return (x-5)**2-10


class ModelPredictor: 
    def __init__(self, X, y):
        self.qbr = QBR(X, y)
        self.qr = QR(X, y)
        self.nn  = QuantileNeuralNetworkOptimized(X, y)
        self.er = ER(X, y)
        self.ebr = EBR(X, y)
    
    def fit_quantile_models(self, alpha:float):
        self.qbr.fit(alpha)
        self.qr.fit(alpha)
        #self.nn.fit(alpha)

    def fit_expectile_models(self):
        for t in range(1, 100):
            t = t/100
            self.ebr.fit(t)
            self.er.fit(t)


    ###Hallamos el VaR con todas las tecnicas###
    def predict_VaR(self, method: str, X_scalar: float, alpha:float): 
        if method == "QBR":
            model = self.qbr
        elif method == "QR": 
            model = self.qr
        elif method == "NN":
            model = self.nn 
        
        return model.predict(alpha, X_scalar)
        
    def predict_ES_scalar(self, method: str, X_scalar:float, alpha: float, predicted_VaR_scalar:float, y):
        if method == "EBR":
            model = self.ebr
        elif method == "ER": 
            model = self.er 

        ##Hay quer hallar el tau que corresponde al VaR
        min_diff = float("inf")
        expectile = None
        tau = None

        for t in range (1, 100):
            t = t/100
            exp = model.predict(t, X_scalar.reshape(1, -1))
            if abs(exp - predicted_VaR_scalar) < min_diff:
                min_diff = abs(exp - predicted_VaR_scalar)
                expectile = exp 
                tau = t
            else: 
                break 
        if tau == 0.5:
            print("Warning: tau is greater than 0.5, this might indicate an issue with the model or data.")
            return 0
        ES = predicted_VaR_scalar + tau/(alpha*(2*tau-1))*(model.predict(0.5, X_scalar.reshape(1, -1))-predicted_VaR_scalar)
        print(f"Predicted ES at {alpha}: {ES} for tau: {tau} and expectile: {expectile} and VaR: {predicted_VaR_scalar} point: {X_scalar}")
        
        return ES

    def predict_ES(self, method: str, X, alpha: float, predicted_VaR:float, y):
        ES = np.zeros(len(X))
        for i in range(len(X)): 
            ES[i] = self.predict_ES_scalar(method, X_scalar = X[i], alpha = alpha, predicted_VaR_scalar = predicted_VaR[i], y = y)
        return ES
    
    



def plot_all(model : ModelPredictor, X, y):
    xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
    y_lower  = model.predict_VaR("QBR", xx, 0.05)
    y_upper  = model.predict_VaR("QBR", xx, 0.95)
    y_med = model.predict_VaR("QR", xx, 0.5)

    y_low_ES = model.predict_ES("EBR", xx, 0.05, y_lower, f(X))
    expectile = model.ebr.predict(0.05, xx)
    expectile2 = model.ebr.predict(0.1, xx)
    expectile3 = model.ebr.predict(0.2, xx)
    expectile4 = model.ebr.predict(0.5, xx)

    fig = plt.figure(figsize=(50, 50))
    plt.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(8x)-2$")
    plt.plot(X, y, "b.", markersize=10, label="Test observations")
    plt.plot(xx, y_med, "r-", label="Predicted median")
    plt.plot(xx, y_upper, "k-")
    plt.plot(xx, y_lower, "k-")
    plt.plot(xx, y_low_ES, "m-", label="Predicted ES lower bound")
    #plt.plot(xx, expectile, "y-", label="Predicted expectile")
    #plt.plot(xx, expectile2, "c-", label="Predicted expectile 0.1")
    #plt.plot(xx, expectile3, "m-", label="Predicted expectile 0.2")
    #plt.plot(xx, expectile4, "orange", label="Predicted expectile 0.5")
    plt.fill_between(
        xx.ravel(), y_lower.ravel(), y_upper, alpha=0.4, label="Predicted 90% interval"
    )
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.ylim(-25, 25)
    plt.legend(loc="upper left")
    plt.show()





if __name__ == "__main__":
    rng = np.random.RandomState(42)
    X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
    sigma = 0.5 + X.ravel() / 10
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    y = f(X).ravel()+noise

    modelpredictor = ModelPredictor(X,y)

    modelpredictor.fit_quantile_models(0.05) 
    modelpredictor.fit_quantile_models(0.5) 
    modelpredictor.fit_quantile_models(0.95)
    modelpredictor.fit_expectile_models()

    


    predicted_VaR = modelpredictor.predict_VaR("QBR", X, 0.05)
    print(f"Predicted VaR at 95%: {predicted_VaR}")
    predicted_ES = modelpredictor.predict_ES("EBR", X, 0.05, predicted_VaR, y)
    print(f"Predicted ES at 95%: {predicted_ES}")

    plot_all(modelpredictor, X, y)


    


    


