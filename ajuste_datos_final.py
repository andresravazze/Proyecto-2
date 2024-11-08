# -*- coding: utf-8 -*-
"""Ajuste datos final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13KOaICyid7UWOfVv8hEkaytgK8RfoarV
"""

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from google.colab import drive
drive.mount('/content/gdrive')
data = pd.read_csv('gdrive/My Drive/Analitica/Proyecto 2/bank-full.csv',sep=';') # Remove index_col=0
#LIMPIEZA DE DATOS




# Label encoding for binary categorical columns
binary_columns = ['default', 'housing', 'loan', 'y']
for col in binary_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

data['job'].fillna(data['job'].mode()[0], inplace=True)
data['education'].fillna(data['education'].mode()[0], inplace=True)


# One-hot encoding for multi-category columns
multi_category_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
data = pd.get_dummies(data, columns=multi_category_columns)


# Replace 'unknown' with a more appropriate label or strategy (e.g., 'unknown' -> NaN, then impute or remove)
data.replace('unknown', pd.NA, inplace=True)

# Optionally drop rows with missing values or impute them (here we impute with the mode for simplicity)
data.fillna(data.mode().iloc[0], inplace=True)

# Step 3: Verify the data is ready for analysis
print(data.info())
print(data.head())
print(data.columns)

#EXPLORACION DE DATOS

# Histogramas
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
data['age'].hist(ax=axes[0, 0])
axes[0, 0].set_title('Histograma de Edad')

data['balance'].hist(ax=axes[0, 1])
axes[0, 1].set_title('Histograma de Saldo')

sns.countplot(x='housing', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Distribución de Crédito de Vivienda')

sns.countplot(x='loan', data=data, ax=axes[1, 1])
axes[1, 1].set_title('Distribución de Crédito Personal')
plt.tight_layout()
plt.show()

#Diagrama de Cajas

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
data.boxplot(column='age', ax=axes[0])
axes[0].set_title('Diagrama de Caja de Edad')
data.boxplot(column='balance', ax=axes[1])
axes[1].set_title('Diagrama de Caja de Saldo')
plt.tight_layout()
plt.show()

# Diagramas de violín


# Diagramas de dispersión
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
data.plot(kind='scatter', x='age', y='balance', ax=axes[0])
axes[0].set_title('Edad vs. Saldo')

data.plot(kind='scatter', x='age', y='y', ax=axes[1])
axes[1].set_title('Edad vs. y')
plt.tight_layout()
plt.show()

data.head()

data.isna().sum()

# prompt: hay variables en la data que son dummies podrias pasar si son falsas a o y true a 1

# Assuming 'data' DataFrame is already loaded and processed as in your provided code.

# Identify binary columns (excluding 'y' as it's likely the target variable)
binary_cols = [col for col in data.columns if data[col].isin([0, 1]).all()]
binary_cols = [col for col in binary_cols if col !='y'] #Exclude target variable

# Convert binary columns to numeric (0 and 1)
for col in binary_cols:
    data[col] = data[col].astype(int)

#Alternative method for all columns (including 'y')
#for col in data.columns:
#  if len(data[col].unique()) == 2:
#    data[col]=data[col].map({data[col].unique()[0]: 0, data[col].unique()[1]:1})

print(data.info())
print(data.head())

data.head()

# prompt: descargame el df como unos nuevos datos a la carpeta drive

# Assuming 'data' DataFrame is already created and processed as in your provided code.

# Specify the desired file path in your Google Drive
file_path = '/content/gdrive/My Drive/Analitica/Proyecto 2/processed_bank_data.csv'  # Example path

# Save the DataFrame to the specified file path
data.to_csv(file_path, index=False)  # index=False prevents saving row indices

print(f"DataFrame saved to: {file_path}")

# prompt: sacame el tipo de variables

print(type(data))
data.dtypes