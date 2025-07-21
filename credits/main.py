from ml.expectileboosting import EBR
from ml.quantileboosting import QBR
from ml.neuralnetwork import QuantileNeuralNetwork
from regression.quantileregression import QR 
from regression.expectileregression import ER 
import matplotlib.cm as cm
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time



class ModelPredictor: 
    def __init__(self, X, y):
        self.qbr = QBR(X, y)
        self.qr = QR(X, y)
        self.qnn  = QuantileNeuralNetwork(X, y)
        self.er = ER(X, y)
        self.ebr = EBR(X, y)
    
    def fit_quantile_models(self, alpha:float):
        self.qbr.fit(alpha)
        self.qr.fit(alpha)
        #self.qnn.fit(alpha)

    def fit_expectile_models(self):
        for t in range(1, 20):
            t = t/100
            self.ebr.fit(t)
            self.er.fit(t)
        
        self.ebr.fit(0.5)  # Fitting at tau = 0.5 for EBR
        self.er.fit(0.5)
            
        

    ###Hallamos el VaR con todas las tecnicas###
    def predict_VaR(self, method: str, X_scalar, alpha:float): 
        if method == "QBR":
            return self.qbr.predict(alpha, X_scalar)
        elif method == "QR": 
            return self.qr.predict(alpha, X_scalar)
        elif method == "QNN":
            return self.qnn .predict(alpha, X_scalar)
        
    def predict_ES_scalar(self, method: str, X_scalar:float, alpha: float, predicted_VaR_scalar:float, y):
        if method == "EBR":
            model = self.ebr
        elif method == "ER": 
            model = self.er 

        ##Hay que hallar el tau que corresponde al VaR
        min_diff = float("inf")
        expectile = None
        tau = None

        # Buscamos el tau que minimiza la diferencia entre el VaR predicho y expectil
        for t in range (1, 20):
            t = t/100
            exp = model.predict(t, X_scalar.reshape(1, -1))
            if abs(exp - predicted_VaR_scalar) < min_diff:
                min_diff = abs(exp - predicted_VaR_scalar)
                expectile = exp 
                tau = t
            else: 
                break 
        if tau == 0.5:
            print("Warning: tau is  0.5, this might indicate an issue with the model or data.")
            return 0
        
        ES = predicted_VaR_scalar + tau/(alpha*(2*tau-1))*(model.predict(0.5, X_scalar.reshape(1, -1))-predicted_VaR_scalar)
        return ES


    def predict_ES(self, method: str, X, alpha: float, predicted_VaR:float, y):
        ES = np.zeros(len(X))
        for i in range(len(X)): 
            ES[i] = self.predict_ES_scalar(method, X_scalar = X[i], alpha = alpha, predicted_VaR_scalar = predicted_VaR[i], y = y)
        return ES
    
    def VaR_Score(self, method: str, X, y, alpha: float):
        if method == "QBR":
            return self.qbr.score(alpha, X, y)
        elif method == "QR":
            return self.qr.score(alpha, X, y)
        elif method == "QNN":
            return self.qnn.score(alpha, X, y)
        
    def ES_Score(self, method: str, X, y, alpha: float):
        if method == "EBR":
            model = self.ebr
        elif method == "ER":
            model = self.er
        return model.score(alpha, X, y)

 


if __name__ == "__main__":
    # Campos del archivo cleaned_rental_data.csv:
    # precio: Precio del alquiler (variable objetivo)
    # metros_cuadrados_construidos: Metros cuadrados construidos de la propiedad
    # habitaciones: Número de habitaciones
    # banos: Número de baños
    # planta: Planta en la que se encuentra la propiedad (0 para bajo, -1 para sótano, etc.)
    # latitud: Latitud de la ubicación de la propiedad
    # longitud: Longitud de la ubicación de la propiedad
    # ascensor: Si la propiedad tiene ascensor (1 si sí, 0 si no)
    # obra_nueva: Si la propiedad es de obra nueva (1 si sí, 0 si no)
    # piscina: Si la propiedad tiene piscina (1 si sí, 0 si no)
    # terraza: Si la propiedad tiene terraza (1 si sí, 0 si no)
    # parking: Si la propiedad tiene parking (1 si sí, 0 si no)
    # parking_incluido_en_el_precio: Si el parking está incluido en el precio (1 si sí, 0 si no)
    # aire_acondicionado: Si la propiedad tiene aire acondicionado (1 si sí, 0 si no)
    # trastero: Si la propiedad tiene trastero (1 si sí, 0 si no)
    # jardin: Si la propiedad tiene jardín (1 si sí, 0 si no)
    # planta_was_nan: Si el valor de la planta original era NaN (1 si sí, 0 si no)

    new_house = [73, 3, 1, 5, 40.418395, -3.677125, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]

    #df = pd.read_csv("rental_data/cleaned_rental_data.csv", sep=',', encoding='utf-8-sig')
    #y = df["precio"]
    #X = df.drop(columns=["precio"]).values
##
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)
    #modelpredictor = ModelPredictor(X_scaled,y)
#
    #modelpredictor.fit_quantile_models(0.05)
    #modelpredictor.fit_expectile_models()
#
    quantile_types = ["QBR", "QR", "QNN"]
    expectile_types = ["EBR", "ER"]
    VaR_score = {}
    ES_score = {}
    predictions_for_plot = {}
    predictions_ES_for_plot = {}
#
    #for quantile_type in quantile_types:
    #    predicted_VaR = modelpredictor.predict_VaR(quantile_type, X_scaled, 0.05)
    #    predictions_for_plot[quantile_type] = predicted_VaR
    #    VaR_score[quantile_type] = modelpredictor.VaR_Score(quantile_type, X_scaled, y, 0.05)
#
    #print("VaR Scores:")
    #for quantile_type, score in VaR_score.items():
    #    print(f"{quantile_type}: {score}")
   

    df = pd.read_csv("rental_data/outlierless.csv", sep=',', encoding='utf-8-sig')
    
    y = df["precio"]
    X = df.drop(columns=["precio"]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    modelpredictor = ModelPredictor(X_scaled,y)

    modelpredictor.fit_quantile_models(0.05)
    modelpredictor.fit_expectile_models()

    for quantile_type in quantile_types:
        predicted_VaR = modelpredictor.predict_VaR(quantile_type, X_scaled, 0.05)
        predictions_for_plot[quantile_type] = predicted_VaR
        VaR_score[quantile_type] = modelpredictor.VaR_Score(quantile_type, X_scaled, y, 0.05)


    print("VaR Scores:")
    for quantile_type, score in VaR_score.items():
        print(f"{quantile_type}: {score}")

    # --- Creación de la gráfica ---
    plot_df = pd.DataFrame({'Rent Price': y})
    for model, predictions in predictions_for_plot.items():
        plot_df[f'{model} VaR'] = predictions

    # Ordenar para graficar
    plot_df_sorted = plot_df.sort_values(by='Rent Price', ascending=False).reset_index(drop=True)

    # Configurar figura
    plt.figure(figsize=(15, 8))

    # Línea de precios reales
    plt.plot(
        plot_df_sorted.index,
        plot_df_sorted['Rent Price'],
        label='Precio de Alquiler',
        color='black',
        linewidth=2
    )

    # Colormap para asignar colores distintos automáticamente
    cmap = cm.get_cmap('tab10', len(predictions_for_plot))

    # Pintar cada modelo con color distinto
    for idx, model in enumerate(predictions_for_plot.keys()):
        plt.scatter(
            plot_df_sorted.index,
            plot_df_sorted[f'{model} VaR'],
            label=f'VaR Predicho ({model}, alpha=0.05)',
            color=cmap(idx),
            s=1, #puntos más pequeños 
            alpha=0.7
        )

    plt.title('Comparación de Precios de Viviendas y VaR Predicho por Modelos')
    plt.xlabel('Viviendas (ordenadas por precio)')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True)

    plt.savefig("grafico_var_vs_precio2.png", dpi=300, bbox_inches='tight')
    plt.show()
    # --- Fin de la gráfica ---


    # --- Creación de la gráfica de errores ---
    plot_df = pd.DataFrame({'Rent Price': y})
    for model, predictions in predictions_for_plot.items():
        plot_df[f'{model} VaR'] = predictions

    # Ordenar para graficar
    plot_df_sorted = plot_df.sort_values(by='Rent Price', ascending=False).reset_index(drop=True)

    # Configurar figura
    plt.figure(figsize=(15, 8))

    # Colormap para asignar colores distintos automáticamente
    cmap = cm.get_cmap('tab10', len(predictions_for_plot))

    # Pintar cada modelo con color distinto
    for idx, model in enumerate(predictions_for_plot.keys()):
        error = plot_df_sorted[f'{model} VaR'] - plot_df_sorted['Rent Price']
        plt.scatter(
            plot_df_sorted.index,
            error,
            label=f'Error ({model} - Real)',
            color=cmap(idx),
            s=1,
            alpha=0.7
        )

    plt.title('Error de Predicción vs. Precio de Alquiler Real')
    plt.xlabel('Viviendas (ordenadas por precio)')
    plt.ylabel('Error de Predicción (Predicción - Real)')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0 to see the error bias

    plt.savefig("grafico_errores_prediccion.png", dpi=300, bbox_inches='tight')
    plt.show()
    # --- Fin de la gráfica ---


    ##predecir el precio de new_house
    new_house = np.array(new_house).reshape(1, -1)
    scaled_new_house = scaler.transform(new_house)

    print("Predicted VaR for new house at 0.05:", modelpredictor.predict_VaR("QBR", scaled_new_house, 0.05))
    print("Predicted VaR for new house at 0.05:", modelpredictor.predict_VaR("QR", scaled_new_house, 0.05))
    print("Predicted VaR for new house at 0.05:", modelpredictor.predict_VaR("QNN", scaled_new_house, 0.05))

    print("\n\n--- ES Calculation ---")
    for quantile_type in quantile_types:
        predicted_VaR = modelpredictor.predict_VaR(quantile_type, scaled_new_house, 0.05)
        print(f"\nCalculating ES for VaR from {quantile_type} (VaR = {predicted_VaR}")
        for expectile_type in expectile_types:
            predicted_ES = modelpredictor.predict_ES_scalar(
                method=expectile_type,
                X_scalar=scaled_new_house[0],
                alpha=0.05,
                predicted_VaR_scalar=predicted_VaR[0],
                y=y
            )
            print(f"  - With {expectile_type} and {quantile_type}:  (Predicted ES: {predicted_ES})")


    # --- Calculation of ES for all data points ---
    es_predictions_for_plot = {}
    for quantile_type in quantile_types:
        predicted_VaR = predictions_for_plot[quantile_type]
        for expectile_type in expectile_types:
            print(f"Calculating ES for {quantile_type} and {expectile_type}")
            predicted_ES = modelpredictor.predict_ES(
                method=expectile_type,
                X=X_scaled,
                alpha=0.05,
                predicted_VaR=predicted_VaR,
                y=y
            )
            es_predictions_for_plot[(quantile_type, expectile_type)] = predicted_ES

    # --- Create plot with ES ---
    plot_df = pd.DataFrame({'Rent Price': y})
    for model, predictions in predictions_for_plot.items():
        plot_df[f'{model} VaR'] = predictions

    for (q_model, e_model), predictions in es_predictions_for_plot.items():
        plot_df[f'ES ({q_model}/{e_model})'] = predictions

    # Sort to plot
    plot_df_sorted = plot_df.sort_values(by='Rent Price', ascending=False).reset_index(drop=True)

    # Configure figure
    plt.figure(figsize=(15, 8))

    # Real prices line
    plt.plot(
        plot_df_sorted.index,
        plot_df_sorted['Rent Price'],
        label='Precio de Alquiler',
        color='black',
        linewidth=2
    )


    # Colormap for ES
    cmap_es = cm.get_cmap('viridis', len(es_predictions_for_plot))

    # Plot each ES model
    for idx, ((q_model, e_model), _) in enumerate(es_predictions_for_plot.items()):
        plt.scatter(
            plot_df_sorted.index,
            plot_df_sorted[f'ES ({q_model}/{e_model})'],
            label=f'ES Predicho ({q_model}/{e_model}, alpha=0.05)',
            color=cmap_es(idx),
            s=1, # Make ES points larger to distinguish them
        )

    plt.title('Comparación de Precios de Viviendas y ES Predichos')
    plt.xlabel('Viviendas (ordenadas por precio)')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True)

    plt.savefig("grafico_es_vs_precio.png", dpi=300, bbox_inches='tight')
    plt.show()
    # --- End of ES plot ---
