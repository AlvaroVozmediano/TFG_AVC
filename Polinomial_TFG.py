import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
#%%

# Configuración de la semilla aleatoria para reproducibilidad
random.seed(2017)

# Lectura y preprocesamiento de datos
data = pd.read_csv("C:/Users/alv_1/OneDrive/Escritorio/TFG/solar.csv", sep=';', decimal=',', encoding='utf-8', header=None, dtype=str)
data = data.dropna()
data = data.replace(',', '.', regex=True)
data.columns = ['RS_Entrada', 'TGPi', 'VDC/GFV', 'IDC/GFV', 'VAC/INV', 'IAC/INV', 'PotAC']
data['RS_Entrada'] = pd.to_numeric(data['RS_Entrada'], errors='coerce')
data['TGPi'] = pd.to_numeric(data['TGPi'], errors='coerce')
data['VDC/GFV'] = pd.to_numeric(data['VDC/GFV'], errors='coerce')
data['IDC/GFV'] = pd.to_numeric(data['IDC/GFV'], errors='coerce')
data['VAC/INV'] = pd.to_numeric(data['VAC/INV'], errors='coerce')
data['IAC/INV'] = pd.to_numeric(data['IAC/INV'], errors='coerce')
data['PotAC'] = pd.to_numeric(data['PotAC'], errors='coerce')

YDatai = data.loc[:, 'TGPi']
Xdata = data.drop('TGPi', axis=1)

X_train, X_test, y_train, y_test = train_test_split(Xdata, YDatai, test_size=0.2, random_state=42)

# Escalado de características
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creación y entrenamiento del modelo de regresión polinomial
degree = 4  # Grado del polinomio
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Predicción de la TGPi
y_pred_poly = model_poly.predict(X_test_poly)

# Definición de funciones de métricas de rendimiento
# Error Cuadrático Medio Normalizado
def NMSE(y_true, y_pred_poly):
    return MSE(normalize(y_true), normalize(y_pred_poly))

# Normalización de datos
def normalize(x):
    return (x - min(x))/(max(x)-min(x))

# Error Porcentual Absoluto Medio Simétrico
def sMAPE(y_true, y_pred_poly):
    return np.mean(2 * np.abs(y_true - y_pred_poly) / (np.abs(y_true) + np.abs(y_pred_poly)))

# Error Absoluto Escalado Medio
def MASE(y_true, y_pred_poly,y_train):
    n = y_pred_poly.shape[0]
    d = np.abs(np.diff(y_pred_poly)).sum()/(n-1)
    errors = np.abs(y_true - y_pred_poly )
    return errors.mean()/d

#Error Porcentual Absoluto Medio
def MAPE(y_true, y_pred_poly):
    return np.mean(np.abs((y_true - y_pred_poly) / y_true))

# Error Absoluto Medio
def MAE(y_true, y_pred_poly):
    return metrics.mean_absolute_error(y_true, y_pred_poly)

# Logaritmo del Error Logarítmico Cuadrático Medio
def LMLS(y_true, y_pred_poly):
   residual   = y_pred_poly  - y_true
   result     =  np.mean(np.log(1 + residual**2 / 2))
   return result

# Error Cuadrático Medio 
def MSE(y_true, y_pred_poly):
    return metrics.mean_squared_error(y_true, y_pred_poly)

# Cálculo y presentación de métricas de rendimiento
print("MSE:", MSE(y_test, y_pred_poly))
print("MAE:", MAE(y_test, y_pred_poly))
print("MAPE:", MAPE(y_test, y_pred_poly))
print("sMAPE:", sMAPE(y_test, y_pred_poly))
print("MASE:", MASE(y_test, y_pred_poly, y_train))
print("NMSE:", NMSE(y_test, y_pred_poly))
print("LMLS:", LMLS(y_test, y_pred_poly))

# Gráfico de dispersión de TGPi predicha frente a TGPi real usando Regresión Polinomial
fig = plt.figure(figsize=(8, 6))
plt.title(('Polynomial Regression prediction vs real data'), fontsize=12)
plt.ylabel('TGPi prediction')
plt.xlabel('TGPi real')
plt.tick_params(labelsize=10)
plt.plot(y_test.values[950:1000], 'r', label='Real')
plt.plot(y_pred_poly[950:1000], 'b', label='Predicted')
plt.legend()
plt.show()

# Cálculo y almacenamiento de métricas en el DataFrame results
mse = MSE(y_test, y_pred_poly)
mae = MAE(y_test, y_pred_poly)
lmls = LMLS(y_test, y_pred_poly)
mape = MAPE(y_test, y_pred_poly)
mase = MASE(y_test, y_pred_poly, y_train)
smape = sMAPE(y_test, y_pred_poly)

lista_errores = [mse, mae, lmls, mape, mase, smape]

results = pd.DataFrame()
results['Metrica'] = ['MSE', 'MAE', 'LMLS', 'MAPE', 'MASE', 'sMAPE']
results['Polinomial'] = lista_errores

# Escritura de resultados en un archivo CSV
results.to_csv('C:/Users/alv_1/OneDrive/Escritorio/TFG/resultadosPolinomial.csv', sep=';', decimal=',', index=False)
