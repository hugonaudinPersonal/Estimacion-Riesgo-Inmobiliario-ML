
# Repositorio del Trabajo de Fin de Grado

Este repositorio contiene todos los archivos relacionados con mi Trabajo de Fin de Grado. A continuación, se detalla el contenido de cada una de las carpetas principales.

## Carpeta `credits`

Esta carpeta alberga todo el código fuente desarrollado para el TFG. Su estructura es la siguiente:

- **`main.py`**: Este es el script principal que orquesta todo el proceso de modelado. Carga los datos, entrena los diferentes modelos de regresión cuantilíca y expectílica, genera predicciones y evalúa los resultados.
- **`requirements.txt`**: Este archivo lista todas las dependencias de Python necesarias para ejecutar el código de este proyecto. Para instalar estas dependencias, se puede utilizar el comando `pip install -r requirements.txt`.
- **`rental_data/`**: Esta subcarpeta contiene los datos utilizados en el proyecto, así como scripts para su limpieza y preprocesamiento.
- **`ml/`**: Contiene la implementación de los modelos de machine learning más complejos:
    - `expectileboosting.py`: Implementación del modelo Expectile Boosting.
    - `quantileboosting.py`: Implementación del modelo Quantile Boosting.
    - `neuralnetwork.py`: Implementación de la red neuronal para regresión cuantilíca.
- **`regression/`**: Contiene la implementación de los modelos de regresión más sencillos:
    - `expectileregression.py`: Implementación de la regresión expectílica.
    - `quantileregression.py`: Implementación de la regresión cuantilíca.

## Carpeta `tex`

Esta carpeta contiene todos los archivos relacionados con la memoria del TFG, escrita en LaTeX.

- **`tfg.tex`**: El archivo principal de LaTeX que contiene el cuerpo del documento.
- **`tfg.pdf`**: La versión compilada y final de la memoria del TFG. Este documento contiene la explicación detallada de todo el proyecto, incluyendo la metodología, los resultados y las conclusiones.
- **Imágenes (`.png`, `.jpg`)**: Todas las imágenes y gráficos utilizados en la memoria del TFG se encuentran en esta carpeta.

Para una comprensión completa y detallada del proyecto, se recomienda leer el archivo `tex/tfg.pdf`.
