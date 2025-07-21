# 📈 Técnicas de aprendizaje automático aplicadas a la estimación del Valor a Riesgo y otras medidas de riesgo financiero

¡Hola! 👋 Bienvenido al repositorio de este proyecto. Aquí encontrarás todo el código, los datos y la documentación relacionados con la aplicación de modelos de Machine Learning para la estimación de medidas de riesgo financiero como el **Valor a Riesgo (VaR)** y el **Expected Shortfall (ES)**.

## 🎯 Sobre el Proyecto

Este proyecto explora cómo las técnicas de aprendizaje automático pueden superar las limitaciones de los modelos de regresión lineal tradicionales para estimar con mayor precisión el riesgo en inversiones. El objetivo principal es cuantificar las pérdidas potenciales de una inversión, centrándose en un caso de uso práctico: **la inversión inmobiliaria**.

## ✨ Características Principales

- **Implementación de Modelos**: Se implementan y comparan varios modelos:
  - 🧠 **Redes Neuronales Cuantílicas (QNN)**
  - 🌲 **Boosting Cuantílico (QBR)** y **Boosting Expectílico (EBR)**
  - 📊 **Regresión Cuantil (QR)** y **Regresión Expectil (ER)**
- **Análisis de Datos**: El proyecto incluye scripts para la limpieza y preparación de datos de alquiler de viviendas, incluyendo la eliminación de outliers.
- **Evaluación Comparativa**: Se realiza una evaluación rigurosa de los modelos para determinar cuál ofrece las predicciones más precisas para el VaR y el ES.
- **Caso de Uso Práctico**: Se aplica el mejor modelo para predecir la rentabilidad mínima esperada de una inversión inmobiliaria real, ayudando a la toma de decisiones.

## 📁 Estructura del Repositorio

```
/
├── credits/
│   ├── main.py               # Script principal para ejecutar los modelos
│   ├── requirements.txt      # Dependencias de Python
│   ├── ml/                   # Modelos de Machine Learning (NN, Boosting)
│   ├── regression/           # Modelos de Regresión (Cuantil, Expectil)
│   └── rental_data/          # Datos y scripts de limpieza (R y Python)
│
├── tex/
│   ├── tfg.pdf               # El documento completo del proyecto
│   └── tfg.tex               # Código fuente en LaTeX
│
└── README.md                 # ¡Este archivo!
```

## 🧠 Metodología

El núcleo del proyecto es la comparación de diferentes enfoques para modelar cuantiles (para el VaR) y expectiles (para el ES) de una distribución de precios de alquiler.

1.  **Datos**: Se utilizó una base de datos de ~15,000 anuncios de alquiler en la Comunidad de Madrid, obtenida mediante *web scraping*.
2.  **Modelado**: Se entrenaron modelos de regresión tradicional y de Machine Learning para predecir el precio del alquiler en función de características como la superficie, número de habitaciones, ubicación, etc.
3.  **Estimación de Riesgo**:
    - El **VaR** se estima utilizando los modelos de regresión cuantil (QR, QNN, QBR).
    - El **ES** se estima a partir de los resultados del VaR y los modelos de regresión expectil (ER, EBR).

## 🚀 Puesta en Marcha

Para ejecutar este proyecto en tu máquina local, sigue estos pasos:

### Prerrequisitos

Asegúrate de tener **Python 3** y **R** instalados en tu sistema.

### Instalación

1.  Clona este repositorio:
    ```sh
    git clone <URL-DEL-REPO>
    ```
2.  Navega al directorio del proyecto:
    ```sh
    cd TFG
    ```
3.  Instala las dependencias de Python:
    ```sh
    pip install -r credits/requirements.txt
    ```

### Ejecución

El script principal para entrenar, evaluar y predecir con los modelos es `main.py`.

```sh
python credits/main.py
```

## ✍️ Autor

*   **Hugo Naudín López**

---