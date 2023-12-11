import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
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

model = LinearRegression().fit(X_train, y_train) 
y_pred=model.predict(X_test)

# Definición de funciones de métricas de rendimiento
# Error Cuadrático Medio Normalizado
def NMSE(y_true, y_pred):
    return MSE(normalize(y_true), normalize(y_pred))

# Normalización de datos
def normalize(x):
    return (x - min(x))/(max(x)-min(x))

# Error Porcentual Absoluto Medio Simétrico
def sMAPE(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

# Error Absoluto Escalado Medio
def MASE(y_true, y_pred,y_train):
    n = y_pred.shape[0]
    d = np.abs(np.diff(y_pred)).sum()/(n-1)
    errors = np.abs(y_true - y_pred )
    return errors.mean()/d

#Error Porcentual Absoluto Medio
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# Error Absoluto Medio
def MAE(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)

# Logaritmo del Error Logarítmico Cuadrático Medio
def LMLS(y_true, y_pred):
   residual   = y_pred  - y_true
   result     =  np.mean(np.log(1 + residual**2 / 2))
   return result

# Error Cuadrático Medio 
def MSE(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

# Cálculo de métricas
mse = MSE(y_test, y_pred)
mae = MAE(y_test, y_pred)
lmls = LMLS(y_test, y_pred)
mape = MAPE(y_test, y_pred)
mase = MASE(y_test, y_pred, y_train)
smape = sMAPE(y_test, y_pred)
nmse = NMSE(y_test, y_pred)

# Presentación de métricas de rendimiento
print("MSE:", mse) 
print("MAE:", mae) 
print("MAPE:", mape) 
print("sMAPE:", smape)
print("MASE:", mase)
print("NMSE:", nmse)
print("LMLS:", lmls) 

# Gráfico de dispersión de TGPi predicha frente a TGPi real
fig = plt.figure(figsize=(8,6))
plt.title(('*Model type* prediction vs real data'),fontsize=12)
plt.ylabel('TGPi prediction')
plt.xlabel('TGPi real')
plt.tick_params(labelsize=10)
plt.plot(y_test.values[950:1000], 'r', label='Real')
plt.plot(y_pred[950:1000], 'b', label='Predicted')
plt.show()

#Almacenamiento de métricas en el DataFrame results
lista_errores = [mse, mae, lmls, mape, mase, smape]

results = pd.DataFrame()
results['Metrica'] = ['MSE', 'MAE', 'LMLS', 'MAPE', 'MASE', 'sMAPE']
results['*Model type*'] = lista_errores

# Escritura de resultados en un archivo CSV
results.to_csv('C:/Users/alv_1/OneDrive/Escritorio/TFG/resultadosLineal.csv', sep=';', decimal=',', index=False)
