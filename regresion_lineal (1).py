# -*- coding: utf-8 -*-
"""Regresion_Lineal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LCQ9R38ez-w6r4A-lzKWrPcn6puZUkIn
"""

import pandas as pd

from google.colab import drive
drive.mount('/content/gdrive')

data = pd.read_csv('gdrive/My Drive/Analitica/Proyecto 2/processed_bank_data.csv')

data.head()

data.describe()

print(type(data))
data.dtypes

import seaborn as sns

import matplotlib.pyplot as plt

numeric_columns = data.select_dtypes(include=['number']).columns

for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=False, color='blue')
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

# Crear gráficos de densidad para cada variable numérica
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data[col], color='red', fill=True)
    plt.title(f'Densidad de {col}')
    plt.xlabel(col)
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming data is your DataFrame

# 1. Get all columns that start with 'job_'
job_columns = [col for col in data.columns if col.startswith('job_')]

# 2. Create a new 'job' column by combining the 'job_' columns
#    If a row has a value of 1 in any of the 'job_' columns, it will have
#    the corresponding job name in the new 'job' column. Otherwise, it will be NaN.
data['job'] = data[job_columns].idxmax(axis=1).str.replace('job_', '')

# 3. Replace NaN with 'Unknown' or another appropriate category
data['job'] = data['job'].fillna('Unknown')

# 4. Now you can create the countplot
plt.figure(figsize=(10, 6))
sns.countplot(x='job', data=data)
plt.title('Distribución de Tipos de Trabajo')
plt.xlabel('Tipo de Trabajo')
plt.ylabel('Cantidad')
plt.xticks(rotation=45, ha='right')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Filter data for y == 1
filtered_data = data[data['y'] == 1]

# Create the countplot
plt.figure(figsize=(10, 6))
sns.countplot(x='job', data=filtered_data)
plt.title('Distribución de Tipos de Trabajo cuando y = 1')
plt.xlabel('Tipo de Trabajo')
plt.ylabel('Cantidad')
plt.xticks(rotation=45, ha='right')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create the histogram with zoomed-in x-axis
plt.figure(figsize=(15, 8))
sns.histplot(data=data, x='balance', hue='y', element='step', stat='density', common_norm=False)
plt.title('Distribución de Balance por Valor de y', fontsize=16)
plt.xlabel('Balance', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-1000, 10000)  # Set x-axis limits
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Get all columns that start with 'contact_'
contact_columns = [col for col in data.columns if col.startswith('contact_')]

# 2. Create a new 'contact' column by combining the 'contact_' columns
data['contact'] = data[contact_columns].idxmax(axis=1).str.replace('contact_', '')

# 3. Replace NaN with 'Unknown' or another appropriate category
data['contact'] = data['contact'].fillna('Unknown')

# 4. Create the histogram
plt.figure(figsize=(15, 8))
sns.histplot(data=data, x='contact', hue='y', element='step', stat='density', common_norm=False)
plt.title('Distribución de Contact por Valor de y', fontsize=16)
plt.xlabel('Contact', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
# Select only numerical features for correlation analysis
numerical_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr = numerical_data.corr()

# Create the heatmap
plt.figure(figsize=(16, 16))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Mapa de calor de correlaciones")
plt.show()

for i, col in enumerate(featuresb):
       print(f"Coefficient for {col}: {linreg.coef_[i]}")

import pandas as pd

# Assuming 'featuresb' is your list of predictor variables and 'linreg' is your fitted model
betas = pd.DataFrame({'Variable': featuresb, 'Beta': linreg.coef_})
sorted_betas = betas.sort_values(by='Beta', ascending=False)
print(sorted_betas)

"""Modelo de regresion Lineal"""

# Get all column names
all_columns = data.columns.tolist()

# Exclude 'y' from the list of features
features = [col for col in all_columns if col != 'y']

# dataframe de características
X = data[features]

X.head()

X.shape

# variable de respuesta
y = data['y']

y.head()

# tipos de X y y
print(type(X))
print(type(y))

"""### División entre entrenamiento y prueba

"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# tamaños
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print(X.head())
print(X_train.head())

# cambiando el tamaño del conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# tamaños
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# sin reordenar los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=False)

print(X.head())
print(X_train.head())

# tamaños
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# volviendo al caso en que cambia el tamaño del conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

"""Creaccion del modelo Linear exportacion

"""

from sklearn.linear_model import LinearRegression

# crear el objeto del modelo
linreg = LinearRegression()

# ajustar los parámetros del modelo usando los datos de entrenamiento
linreg.fit(X_train, y_train)

# imprimir coeficientes
print(linreg.intercept_)
print(linreg.coef_)

for i, col in enumerate(features):
    print(f"Coefficient for {col}: {linreg.coef_[i]}")

# coeficientes con nombre de las características
list(zip(features, linreg.coef_))

# prompt: realizame significancia individual a la svariables

import pandas as pd
from google.colab import drive
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats

drive.mount('/content/gdrive')
data = pd.read_csv('gdrive/My Drive/Analitica/Proyecto 2/processed_bank_data.csv')

# ... (rest of your existing code)

# Statistical significance of features in the linear regression model
# Calculate p-values for each feature
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Calculate p-values using statsmodels
import statsmodels.api as sm
X2 = sm.add_constant(X_train) # adding a constant
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

# Alternatively, use the following method to get p-values
# Note this method can be numerically unstable for highly correlated features.
from sklearn.feature_selection import f_regression
f_statistic, p_values = f_regression(X_train, y_train)

significance_df = pd.DataFrame({'Feature': features, 'p-value': p_values})
significance_df = significance_df.sort_values(by='p-value')
print(significance_df)

"""###Predicciones usando los datos de prueba"""

y_pred = linreg.predict(X_test)

"""### Evaluar el modelo

**Error absoluto medio**:

$$\text{MAE} = \frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$

**Error cuadrado medio**:
$$\text{MSE} = \frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$

**Raíz del Error cuadrado medio**:
$$\text{RMSE} = \sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
"""

from sklearn import metrics

import numpy as np

# mean absolute error
MAE = metrics.mean_absolute_error(y_test, y_pred)

# mean squared error
MSE = metrics.mean_squared_error(y_test, y_pred)

# root mean squared error
RMSE = np.sqrt(MSE)

print("MAE: ", MAE)
print("MSE: ", MSE)
print("RMSE: ", RMSE)

import pandas as pd

# 1. Select 'month' columns:
month_cols = [col for col in data.columns if 'month' in col]

# 2. Sum 'month' columns and create 'total_months':
data['contacted'] = data[month_cols].sum(axis=1)

# 3. Drop unwanted columns, including individual 'month' columns:
columns_to_drop = ['previous', 'education_unknown', 'education_secondary', 'marital_single'] + month_cols
new_df = data.drop(columns=columns_to_drop)

# 4. (Optional) Display the new DataFrame:
new_df.head()

"""### Selección de variables - Modelo con otro subconjunto de variables"""

# Get all column names
new_all_columns = new_df.columns.tolist()

# Exclude 'y' from the list of features
featuresb = [col for col in new_all_columns if col != 'y']

X = data[featuresb]
X.head()
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print("MAE: ", MAE)
print("MSE: ", MSE)
print("RMSE: ", RMSE)

"""### Validación cruzada"""

from sklearn.model_selection import cross_val_score

# usar MSE - error cuadrático medio
scores = cross_val_score(linreg, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = - scores
print(mse_scores)

"""## Ahora usando statsmodels"""

import statsmodels.api as sm

features = [col for col in all_columns if col != 'y']

X = data[features]
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# agregar constante explíticamente
X_train = sm.add_constant(X_train)

# regresión usando mínimos cuadrados ordinarios (ordinary least squares - OLS)
model = sm.OLS(y_train, X_train).fit()

# resumen de resultados
print(model.summary())

"""### El segundo modelo, sin Newspaper"""

X = data[featuresb]
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# agregar constante explíticamente
X_train = sm.add_constant(X_train)

# regresión usando mínimos cuadrados ordinarios (ordinary least squares - OLS)
model = sm.OLS(y_train, X_train).fit()

# resumen de resultados
print(model.summary())

"""### Determinar la influencia de las observaciones"""

#fig = sm.graphics.influence_plot(model, criterion="cooks", size=8)

"""### Determinar puntos de alta influencia con distancia de Cook y umbral $4/n$"""

# disntacia de Cook
model_cooksd = model.get_influence().cooks_distance[0]

# get length of df to obtain n
n = X_train.shape[0]

# umbral
critical_d = 4/n
print('Umbral con distancia de Cook:', critical_d)

# puntos que podrían ser ourliers con alta influencia
out_d = model_cooksd > critical_d

print(X_train.index[out_d], "\n", model_cooksd[out_d])

X_train[out_d]

y_train[out_d]

# prompt: descargame el new_df como unos nuevos datos a la carpeta drive

new_df.to_csv('/content/gdrive/My Drive/Analitica/Proyecto 2/new_data.csv')
# Specify the desired file path in your Google Drive
file_path = '/content/gdrive/My Drive/Analitica/Proyecto 2/new_data.csv'  # Example path

# Save the DataFrame to the specified file path
new_df.to_csv(file_path, index=False)  # index=False prevents saving row indices

print(f"DataFrame saved to: {file_path}")

new_df.head()