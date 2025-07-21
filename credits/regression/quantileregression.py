from sklearn.model_selection import KFold
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_pinball_loss




class QR:
    def __init__(self, X, y ):
        self.X_train, self.X_test,self. y_train, self.y_test = train_test_split(X, y, random_state=0)
        self.models = {}
        self.predictions_train = {}
        self.predictions_test = {}

        
    def neg_mean_pinball_loss_scorer(self, alpha):
        return make_scorer(mean_pinball_loss, alpha=alpha,greater_is_better=False, )
    
    #Entrenamiento del modelo
    def fit(self, alpha: float):
        print(f"---------------------------------------Training model for alpha: {alpha} -----------------------------------")

        param_grid = {
            'alpha': [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],  # Regularización L1
            'solver': ['highs-ds', 'highs-ipm', 'highs'],     # solvers
            'fit_intercept': [True, False]                     #intercepto
        }
        model = QuantileRegressor(quantile=alpha)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

    #Busqueda de hiperparámetros con validación cruzada
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=kf,
            scoring=self.neg_mean_pinball_loss_scorer(alpha),
            n_jobs=-1,
            verbose=1,  # Para ver el progreso
            return_train_score=True
        )
        
        # Entrenar con búsqueda de hiperparámetros
        print("Buscando mejores hiperparámetros...")
        grid_search.fit(self.X_train, self.y_train)
        
        # Guardar el mejor modelo
        self.models[alpha] = grid_search.best_estimator_


    def predict(self, alpha, X):
        prediction = self.models[alpha].predict(X)
        return prediction
    
    def score(self, alpha: float, X, y):
        prediction = self.predict(alpha, X)
        score = mean_pinball_loss(y, prediction, alpha=alpha)
        return score


 

 
        
