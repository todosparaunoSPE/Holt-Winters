# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:55:48 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Función para cargar y procesar el archivo CSV
@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {e}")
        return None

# Función para dividir los datos en entrenamiento y prueba
def dividir_datos(series, test_size=0.2):
    n = len(series)
    n_test = int(n * test_size)
    train = series[:-n_test]
    test = series[-n_test:]
    return train, test

# Función para aplicar el modelo de Holt Winters y generar el pronóstico
def aplicar_holt_winters(series, seasonal_periods=12, alpha=0.3, beta=0.2, gamma=0.1):
    train, test = dividir_datos(series, test_size=0.2)
    modelo = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
    resultado = modelo.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
    pronostico = resultado.forecast(len(test))
    y_pred = resultado.fittedvalues
    mse = mean_squared_error(test, pronostico)
    mae = mean_absolute_error(test, pronostico)
    return pronostico, mse, mae, y_pred, test

# Función para aplicar el modelo ARIMA y generar el pronóstico
def aplicar_arima(series, order=(5, 1, 0)):
    train, test = dividir_datos(series, test_size=0.2)
    modelo = ARIMA(train, order=order)
    resultado = modelo.fit()
    pronostico = resultado.forecast(steps=len(test))
    y_pred = resultado.predict(start=0, end=len(train)-1)
    mse = mean_squared_error(test, pronostico)
    mae = mean_absolute_error(test, pronostico)
    return pronostico, mse, mae, y_pred, test

# Función para probar varios modelos y obtener pronósticos y errores
def probar_modelos(series, modelo_seleccionado):
    pronosticos = {}
    try:
        if modelo_seleccionado == 'Holt Winters':
            seasonal_periods = st.sidebar.slider('Período estacional', min_value=1, max_value=24, value=12)
            alpha = st.sidebar.slider('Alpha (Suavizado nivel)', min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            beta = st.sidebar.slider('Beta (Suavizado pendiente)', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            gamma = st.sidebar.slider('Gamma (Suavizado estacional)', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
            pronostico, mse, mae, y_pred, test = aplicar_holt_winters(series, seasonal_periods, alpha, beta, gamma)
            pronosticos['Holt Winters'] = (pronostico, mse, mae, y_pred, test)
        elif modelo_seleccionado == 'ARIMA':
            p = st.sidebar.slider('AR (p)', min_value=0, max_value=10, value=5, step=1)
            d = st.sidebar.slider('Integración (d)', min_value=0, max_value=2, value=1, step=1)
            q = st.sidebar.slider('MA (q)', min_value=0, max_value=10, value=0, step=1)
            order = (p, d, q)
            pronostico, mse, mae, y_pred, test = aplicar_arima(series, order)
            pronosticos['ARIMA'] = (pronostico, mse, mae, y_pred, test)
    except Exception as e:
        st.error(f"Error al probar modelos: {e}")
    return pronosticos

# Función para graficar los resultados de forma interactiva
def grafico_interactivo(series, pronosticos):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(series.index, series, label='Datos reales')
        
        for modelo, (pronostico, _, _, y_pred, test) in pronosticos.items():
            ax.plot(y_pred.index, y_pred, label=f'Ajuste del modelo {modelo}', linestyle='--')
            ax.plot(test.index, test, label=f'Datos de prueba {modelo}', linestyle='--')
            ax.plot(pronostico.index, pronostico, label=f'Pronóstico {modelo}', linestyle='--')
        
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valores')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al crear el gráfico interactivo: {e}")

# Configuración de la aplicación Streamlit
st.title('Aplicación de Modelos de Series Temporales')

# Añadir resumen en la sidebar
st.sidebar.markdown("""
    ## Resumen
    El código crea una aplicación web interactiva donde los usuarios pueden cargar datos de series temporales,
    seleccionar y ajustar modelos de pronóstico (Holt Winters o ARIMA), ver las predicciones y errores del modelo,
    y comparar los resultados en un gráfico interactivo. Además, incluye un aviso de copyright en la parte inferior
    de la página.
""")

# Cargar archivo CSV
archivo = st.file_uploader("Subir archivo CSV", type=['csv'])

if archivo is not None:
    datos = cargar_datos(archivo)
    
    if datos is not None:
        st.subheader('Datos cargados:')
        st.write(datos)
        
        st.subheader('Columnas disponibles en los datos:')
        st.write(datos.columns.tolist())
        
        columnas = datos.columns.tolist()
        fecha_col = st.selectbox('Seleccionar columna de fecha', columnas)
        data_col = st.selectbox('Seleccionar columna de datos', columnas)
        
        st.write(f"Columna de fecha seleccionada: {fecha_col}")
        st.write(f"Columna de datos seleccionada: {data_col}")
        st.write(f"Columnas presentes en el DataFrame: {datos.columns.tolist()}")
        
        try:
            datos[fecha_col] = pd.to_datetime(datos[fecha_col], dayfirst=True)
            datos = datos.set_index(fecha_col)
        except Exception as e:
            st.error(f"Error al convertir la columna de fecha: {e}")
        
        if data_col not in datos.columns:
            st.error(f"La columna seleccionada para los datos no existe: {data_col}")
        else:
            series = datos[data_col]
            modelos_seleccionados = st.sidebar.multiselect('Seleccionar modelos', ['Holt Winters', 'ARIMA'], default=['Holt Winters'])
            pronosticos = {}
            for modelo in modelos_seleccionados:
                pronosticos.update(probar_modelos(series, modelo))
            
            if pronosticos:
                st.subheader('Comparación de Modelos:')
                for modelo, (pronostico, mse, mae, _, _) in pronosticos.items():
                    st.write(f'Modelo: {modelo}')
                    st.write(f'Error cuadrático medio (MSE): {mse:.2f}')
                    st.write(f'Error absoluto medio (MAE): {mae:.2f}')
                    st.write(f'Pronóstico:')
                    st.write(pronostico)
                    st.write('---')
                
                st.subheader('Gráfico interactivo de resultados:')
                grafico_interactivo(series, pronosticos)

# Añadir aviso de copyright
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        &copy; 2024 Tu Nombre o Tu Compañía. Todos los derechos reservados.
    </div>
    """,
    unsafe_allow_html=True
)