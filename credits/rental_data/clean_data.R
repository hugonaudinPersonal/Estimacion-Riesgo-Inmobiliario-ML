# Cargar el dataset
data <- read.csv("credits/rental_data/cleaned_rental_data.csv")
# Lista de variables numéricas
vars <- c("precio", "metros_cuadrados_construidos", "habitaciones", 
          "banos", "planta", "latitud", "longitud")

# Función para contar outliers con criterio de IQR
count_outliers <- function(x) {
  x <- na.omit(x)
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  upper <- Q3 + 1.5 * IQR
  sum(x > upper)
}

# Función para detectar outliers (devuelve vector lógico)
get_outliers_logical <- function(x) {
  x <- na.omit(x)
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  upper <- Q3 + 1.5 * IQR
  return(x > upper)
}


# Aplicar a cada variable del data_sample
outlier_counts <- sapply(vars, function(var) {
  count_outliers(data[[var]])
})

# Mostrar resultados
outlier_counts


# Función auxiliar para alinear con el vector original (por si hay NAs)
align_outliers <- function(outliers_logical, original_vector) {
  full <- rep(NA, length(original_vector))
  full[!is.na(original_vector)] <- outliers_logical
  return(full)
}

# Detectar outliers
precio_outliers <- align_outliers(get_outliers_logical(data$precio), data$precio)


# Eliminar esas filas del dataset
data_sample_clean <- data[!(precio_outliers), ]

# Guardar el dataset limpio en un archivo CSV
write.csv(data_sample_clean, "outlierless.csv", row.names = FALSE)

outlierless <- read.csv("credits/rental_data/outlierless.csv")

