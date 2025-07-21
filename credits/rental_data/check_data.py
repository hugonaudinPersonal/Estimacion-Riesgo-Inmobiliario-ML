import pandas as pd
import glob
import unicodedata
import re
import numpy as np

def clean_col_names(df):
    """
    Cleans and standardizes the column names of a DataFrame.
    - Converts to lowercase.
    - Removes accents and special characters.
    - Replaces spaces and other separators with underscores.
    - Removes parentheses and the content within them.
    """
    new_columns = {}
    for col in df.columns:
        # Remove content in parentheses
        clean_col = re.sub(r'\s*\(.*\)\s*', '', col)
        # Normalize to remove accents
        clean_col = ''.join(c for c in unicodedata.normalize('NFD', clean_col) if unicodedata.category(c) != 'Mn')
        # To lower case
        clean_col = clean_col.lower()
        # Replace spaces and hyphens with underscores
        clean_col = re.sub(r'[\s-]+', '_', clean_col)
        # Remove any other non-alphanumeric characters (except underscore)
        clean_col = re.sub(r'[^a-z0-9_]', '', clean_col)
        new_columns[col] = clean_col
    df = df.rename(columns=new_columns)
    return df

# Get all csv files in the current directory
all_files = glob.glob("*.csv")

# Read and concatenate all files
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, sep=';', encoding='utf-8-sig')
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

# Clean column names
frame = clean_col_names(frame)

# Clean 'metros_cuadrados_construidos' column
frame['metros_cuadrados_construidos'] = frame['metros_cuadrados_construidos'].astype(str).str.replace(r'[^0-9]', '', regex=True)
frame['metros_cuadrados_construidos'] = pd.to_numeric(frame['metros_cuadrados_construidos'], errors='coerce').fillna(0).astype(int)

# Clean latitude and longitude
for col in ['latitud', 'longitud']:
    frame[col] = frame[col].str.replace(',', '.').astype(float)

# Convert boolean columns
for col in ['ascensor', 'obra_nueva', 'piscina', 'terraza', 'parking', 'parking_incluido_en_el_precio', 'aire_acondicionado', 'trastero', 'jardin']:
    frame[col] = frame[col].apply(lambda x: 1 if x == 'Sí' else 0)

# Clean 'planta' column
frame['planta'] = frame['planta'].replace('Bajo', '0')
frame['planta'] = frame['planta'].replace('Entreplanta', '0')
frame['planta'] = frame['planta'].replace('-', np.nan)
frame['planta'] = pd.to_numeric(frame['planta'], errors='coerce')



# Handle NaNs: fill with 0 and create a boolean column indicating original NaN presence
for col in frame.columns:
    if frame[col].isnull().any():
        frame[f'{col}_was_nan'] = frame[col].isnull().astype(int)
        frame[col] = frame[col].fillna(0)

# Show results
print("Cleaned column names:")
print(frame.columns)

print("\nFirst 5 rows of the dataframe:")
print(frame.head())

print("\nDataFrame Info:")
frame.info()

# Save the cleaned DataFrame to a new CSV file
frame.to_csv('cleaned_rental_data.csv', index=False, encoding='utf-8-sig')
print("\nCleaned data saved to cleaned_rental_data.csv")

nans_por_columna = frame.isna().sum()
nans_presentes = nans_por_columna[nans_por_columna > 0]

if not nans_presentes.empty:
    print("\nColumnas con valores NaN y su cantidad:")
    print(nans_presentes)
else:
    print("\nNo hay valores NaN en el DataFrame.")