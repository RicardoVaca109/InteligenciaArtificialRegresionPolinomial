# Polynomial Regression con nuevo dataset

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el nuevo dataset
dataset = pd.read_csv('Study_Scores.csv')
X = dataset[['HorasEstudio']].values      # Ahora la columna se llama 'HorasEstudio'
y = dataset['Calificacion'].values        # y la variable objetivo es 'Calificacion'

# Entrenar modelo de regresión lineal
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Entrenar modelo de regresión polinómica (grado 4)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizar resultados de Regresión Lineal
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Regresión Lineal vs Datos')
plt.xlabel('Horas de Estudio')
plt.ylabel('Calificación')
plt.show()

# Visualizar resultados de Regresión Polinómica
plt.scatter(X, y, color='red')
# plt.plot(X, lin_reg_2.predict(poly_reg.transform(X)), color='blue')
# 1) Genera un grid ordenado
X_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
# 2) Predice sobre ese grid
y_grid = lin_reg_2.predict(poly_reg.transform(X_grid))
# 3) Dibuja la curva
plt.plot(X_grid, y_grid, color='blue', linewidth=2)

plt.title('Regresión Polinómica (grado 4)')
plt.xlabel('Horas de Estudio')
plt.ylabel('Calificación')
plt.show()

# Gráfica con alta resolución para la curva polinómica
X_grid = np.arange(min(X)[0], max(X)[0], 0.1).reshape(-1, 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.transform(X_grid)), color='blue')
plt.title('Regresión Polinómica (suavizado)')
plt.xlabel('Horas de Estudio')
plt.ylabel('Calificación')
plt.show()

# Predicciones de ejemplo
print("Predicción lineal para 6.5 horas:", lin_reg.predict([[6.5]]))
print("Predicción polinómica para 6.5 horas:", lin_reg_2.predict(poly_reg.transform([[6.5]])))
