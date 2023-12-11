##"C:\Users\alv_1\AppData\Local\Programs\Python\Python38\python.exe"
# Importación de librerías
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.pipeline import Pipeline
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


# Definición de funciones de métricas de rendimiento
# Error Cuadrático Medio Normalizado
def NMSE(y_true, y_pred_MLP):
    return MSE(normalize(y_true), normalize(y_pred_MLP))

# Normalización de datos
def normalize(x):
    return (x - min(x))/(max(x)-min(x))

# Error Porcentual Absoluto Medio Simétrico
def sMAPE(y_true, y_pred_MLP):
    return np.mean(2 * np.abs(y_true - y_pred_MLP) / (np.abs(y_true) + np.abs(y_pred_MLP)))

# Error Absoluto Escalado Medio
def MASE(y_true, y_pred_MLP,y_train):
    n = y_pred_MLP.shape[0]
    d = np.abs(np.diff(y_pred_MLP)).sum()/(n-1)
    errors = np.abs(y_true - y_pred_MLP )
    return errors.mean()/d

# Error Porcentual Absoluto Medio
def MAPE(y_true, y_pred_MLP):
    return np.mean(np.abs((y_true - y_pred_MLP) / y_true))

# Error Absoluto Medio
def MAE(y_true, y_pred_MLP):
    return metrics.mean_absolute_error(y_true, y_pred_MLP)

# Logaritmo del Error Logarítmico Cuadrático Medio
def LMLS(y_true, y_pred_MLP):
   residual   = y_pred_MLP  - y_true
   result     =  np.mean(np.log(1 + residual**2 / 2))

   return result

# Error Cuadrático Medio 
def MSE(y_true, y_pred_MLP):
    return metrics.mean_squared_error(y_true, y_pred_MLP)


# Escalado de características
std_scaler = MinMaxScaler()
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

# Creación y entrenamiento del modelo MLPRegressor
reg = MLPRegressor(solver='adam', hidden_layer_sizes=(20,),
                   max_iter=1000, shuffle=True, random_state=1, activation='tanh')
reg.fit(X_train_scaled, y_train)
pipelineREG = Pipeline(memory=None,
         steps=[('scaler', std_scaler),
                ('reg', reg)])

# Predicción de la TGPi
y_pred_MLP = reg.predict(X_test_scaled)

# Creación del DataFrame de resultados
from sklearn.model_selection import train_test_split
results = pd.DataFrame()
pred = pd.DataFrame()


# Cálculo y presentación de métricas de rendimiento
print("MSE:", MSE(y_test, y_pred_MLP)) 
print("MAE:", MAE(y_test, y_pred_MLP)) 
print("MAPE:", MAPE(y_test, y_pred_MLP)) 
print("sMAPE:", sMAPE(y_test, y_pred_MLP))
print("MASE:", MASE(y_test, y_pred_MLP, y_train))
print("NMSE:", NMSE(y_test, y_pred_MLP)) 
print("LMLS:", LMLS(y_test, y_pred_MLP)) 

# Gráfico de dispersión de TGPi predicha frente a TGPi real
fig = plt.figure(figsize=(8,6))
plt.title(('MLP prediction vs real data'),fontsize=12)
plt.ylabel('TGPi prediction')
plt.xlabel('TGPi real')
plt.tick_params(labelsize=10)
plt.plot(y_test.values[300:350], 'r', label='Real')
plt.plot(y_pred_MLP[300:350], 'b', label='Predicted')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.title(('MLP prediction vs real data'),fontsize=12)
plt.ylabel('TGPi prediction')
plt.xlabel('TGPi real')
plt.tick_params(labelsize=10)
plt.plot(y_test.values[550:600], 'r', label='Real')
plt.plot(y_pred_MLP[550:600], 'b', label='Predicted')
plt.show()

fig = plt.figure(figsize=(8,6))
plt.title(('MLP prediction vs real data'),fontsize=12)
plt.ylabel('TGPi prediction')
plt.xlabel('TGPi real')
plt.tick_params(labelsize=10)
plt.plot(y_test.values[950:1000], 'r', label='Real')
plt.plot(y_pred_MLP[950:1000], 'b', label='Predicted')
plt.show()

# Cálculo y almacenamiento de métricas en el DataFrame results
mse = MSE(y_test, y_pred_MLP)
mae = MAE(y_test, y_pred_MLP)
lmls = LMLS(y_test, y_pred_MLP)
mape = MAPE(y_test, y_pred_MLP)
mase = MASE(y_test, y_pred_MLP, y_train)
smape = sMAPE(y_test, y_pred_MLP)

lista_errores = [mse, mae, lmls, mape, mase, smape]

results = pd.DataFrame()
results['Metrica'] = ['MSE', 'MAE', 'LMLS', 'MAPE', 'MASE', 'sMAPE']
results['MLP'] = lista_errores

# Escritura de resultados en un archivo CSV
results.to_csv('C:/Users/alv_1/OneDrive/Escritorio/TFG/resultadosMLP.csv', sep=';', decimal=',', index=False)