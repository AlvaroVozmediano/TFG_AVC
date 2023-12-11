##"C:\Users\alv_1\AppData\Local\Programs\Python\Python38\python.exe"
# Importación de librerías 
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import warnings
#%%

# Configuración de la semilla aleatoria para reproducibilidad
random.seed(2017)

# Lectura y preprocesamiento de datos
data = pd.read_csv("C:/Users/alv_1/OneDrive/Escritorio/TFG/solar.csv",sep=';', 
                   decimal = ',', encoding = 'utf-8',header=None, dtype=str)
data = data.dropna()
data = data.replace(',', '.', regex=True)
data.columns=['RS_Entrada','TGPi','VDC/GFV','IDC/GFV','VAC/INV','IAC/INV','PotAC']
data['RS_Entrada'] = pd.to_numeric(data['RS_Entrada'], errors='coerce')
data['TGPi'] = pd.to_numeric(data['TGPi'], errors='coerce')
data['VDC/GFV'] = pd.to_numeric(data['VDC/GFV'], errors='coerce')
data['IDC/GFV'] = pd.to_numeric(data['IDC/GFV'], errors='coerce')
data['VAC/INV'] = pd.to_numeric(data['VAC/INV'], errors='coerce')
data['IAC/INV'] = pd.to_numeric(data['IAC/INV'], errors='coerce')
data['PotAC'] = pd.to_numeric(data['PotAC'], errors='coerce')

# División de datos en entrenamiento y prueba
YDatai = data.loc[ :, 'TGPi' ]
Xdata = data.drop('TGPi',axis=1)

X_train, X_test, y_train, y_test = train_test_split(Xdata, YDatai, test_size=0.2, random_state=42)

# Escalado de características 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creación y entrenamiento del modelo SVR
reg_svr = SVR(kernel='rbf', C=1, epsilon=0.2)
reg_svr.fit(X_train_scaled, y_train)

# Predicción de la TGPi
y_pred_svr = reg_svr.predict(X_test_scaled)

# Definición de funciones de métricas de rendimiento
# Error Cuadrático Medio Normalizado
def NMSE(y_true, y_pred_svr):
    return MSE(normalize(y_true), normalize(y_pred_svr))

# Normalización de datos
def normalize(x):
    return (x - min(x))/(max(x)-min(x))

# Error Porcentual Absoluto Medio Simétrico
def sMAPE(y_true, y_pred_svr):
    return np.mean(2 * np.abs(y_true - y_pred_svr) / (np.abs(y_true) + np.abs(y_pred_svr)))

# Error Absoluto Escalado Medio
def MASE(y_true, y_pred_svr,y_train):
    n = y_pred_svr.shape[0]
    d = np.abs(np.diff(y_pred_svr)).sum()/(n-1)
    errors = np.abs(y_true - y_pred_svr )
    return errors.mean()/d

# Error Porcentual Absoluto Medio
def MAPE(y_true, y_pred_svr):
    return np.mean(np.abs((y_true - y_pred_svr) / y_true))

# Error Absoluto Medio
def MAE(y_true, y_pred_svr):
    return metrics.mean_absolute_error(y_true, y_pred_svr)

# Logaritmo del Error Logarítmico Cuadrático Medio
def LMLS(y_true, y_pred_svr):
   residual   = y_pred_svr  - y_true
   result     =  np.mean(np.log(1 + residual**2 / 2))

   return result
# Error Cuadrático Medio 
def MSE(y_true, y_pred_svr):
    return metrics.mean_squared_error(y_true, y_pred_svr)

# Cálculo y presentación de métricas de rendimiento
print("MSE:", MSE(y_test, y_pred_svr)) 
print("MAE:", MAE(y_test, y_pred_svr)) 
print("MAPE:", MAPE(y_test, y_pred_svr)) 
print("sMAPE:", sMAPE(y_test, y_pred_svr))
print("MASE:", MASE(y_test, y_pred_svr, y_train))
print("NMSE:", NMSE(y_test, y_pred_svr)) 
print("LMLS:", LMLS(y_test, y_pred_svr)) 

# Gráfico de dispersión de TGPi predicha frente a TGPi real usando SVR

fig = plt.figure(figsize=(8,6))
plt.title(('SVR prediction vs real data'),fontsize=12)
plt.ylabel('TGPi prediction')
plt.xlabel('TGPi real')
plt.tick_params(labelsize=10)
plt.plot(y_test.values[300:350], 'r', label='Real')
plt.plot(y_pred_svr[300:350], 'b', label='Predicted')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.title(('SVR prediction vs real data'),fontsize=12)
plt.ylabel('TGPi prediction')
plt.xlabel('TGPi real')
plt.tick_params(labelsize=10)
plt.plot(y_test.values[550:600], 'r', label='Real')
plt.plot(y_pred_svr[550:600], 'b', label='Predicted')
plt.show()


fig = plt.figure(figsize=(8,6))
plt.title(('SVR prediction vs real data'),fontsize=12)
plt.ylabel('TGPi prediction')
plt.xlabel('TGPi real')
plt.tick_params(labelsize=10)
plt.plot(y_test.values[950:1000], 'r', label='Real')
plt.plot(y_pred_svr[950:1000], 'b', label='Predicted')
plt.show()

# Cálculo y almacenamiento de métricas en el DataFrame results
mse = MSE(y_test, y_pred_svr)
mae = MAE(y_test, y_pred_svr)
lmls = LMLS(y_test, y_pred_svr)
mape = MAPE(y_test, y_pred_svr)
mase = MASE(y_test, y_pred_svr, y_train)
smape = sMAPE(y_test, y_pred_svr)

lista_errores = [mse, mae, lmls, mape, mase, smape]

results = pd.DataFrame()
results['Metrica'] = ['MSE', 'MAE', 'LMLS', 'MAPE', 'MASE', 'sMAPE']
results['SVR'] = lista_errores

# Escritura de resultados en un archivo CSV
results.to_csv('C:/Users/alv_1/OneDrive/Escritorio/TFG/resultadosSVR.csv', sep=';', decimal=',', index=False)
