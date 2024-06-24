# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:25:35 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Descripción general de la aplicación
st.sidebar.markdown("""
    ## Análisis de Series Temporales
    
    El código integra capacidades de análisis de series temporales mediante modelos como Holt Winters y ARIMA,
    y proporciona una interfaz interactiva para cargar datos, ajustar modelos, visualizar resultados y realizar
    pronósticos futuros, todo implementado en una aplicación web usando Streamlit.
""")


# Función para cargar y procesar el archivo CSV
@st.cache_data()
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo)
        return df.copy()  # Ensure a copy is returned to avoid mutations
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {e}")
        return None

# Función para dividir los datos en entrenamiento y prueba
def dividir_datos(series, test_size=0.2):
    n = len(series)
    n_test = int(n * test_size)
    train = series[:-n_test]
    test = series[-(n_test):]
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

# Función para probar modelos y obtener pronósticos, ajustes y errores
def probar_modelos(series, modelos):
    ajustes = {}
    pronosticos = {}
    errores = {}

    for modelo in modelos:
        if modelo == 'Holt Winters':
            seasonal_periods = st.sidebar.slider('Período estacional', min_value=1, max_value=24, value=12)
            alpha = st.sidebar.slider('Alpha (Suavizado nivel)', min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            beta = st.sidebar.slider('Beta (Suavizado pendiente)', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            gamma = st.sidebar.slider('Gamma (Suavizado estacional)', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
            pronostico, mse, mae, y_pred, test = aplicar_holt_winters(series, seasonal_periods, alpha, beta, gamma)
            ajustes['Holt Winters'] = (y_pred, test)
            pronosticos['Pronóstico Holt Winters'] = pronostico
            errores[f'MSE {modelo}'] = mse
            errores[f'MAE {modelo}'] = mae
        elif modelo == 'ARIMA':
            p = st.sidebar.slider('AR (p)', min_value=0, max_value=10, value=5, step=1)
            d = st.sidebar.slider('Integración (d)', min_value=0, max_value=2, value=1, step=1)
            q = st.sidebar.slider('MA (q)', min_value=0, max_value=10, value=0, step=1)
            order = (p, d, q)
            pronostico, mse, mae, y_pred, test = aplicar_arima(series, order)
            ajustes['ARIMA'] = (y_pred, test)
            pronosticos['Pronóstico ARIMA'] = pronostico
            errores[f'MSE {modelo}'] = mse
            errores[f'MAE {modelo}'] = mae

    return ajustes, pronosticos, errores

# Función para graficar los resultados de forma interactiva
def grafico_interactivo(series, ajustes, pronosticos, errores, tipo_grafico):
    if tipo_grafico == 'Ajustes de Modelos':
        for modelo, (y_pred, test) in ajustes.items():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines', name=f'Ajuste del modelo {modelo}', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name=f'Datos de prueba {modelo}', line=dict(dash='dot')))
            fig.update_layout(
                title=f'Ajustes del modelo {modelo}',
                xaxis_title='Fecha',
                yaxis_title='Valores',
                legend_title='Series',
                hovermode='x unified',
                xaxis_rangeslider_visible=True,
                width=900,  # Ancho del gráfico
                height=600,  # Alto del gráfico
            )
            st.plotly_chart(fig)
            
            st.subheader(f'Errores del modelo {modelo}')
            st.write(f'MSE: {errores[f"MSE {modelo}"]}')
            st.write(f'MAE: {errores[f"MAE {modelo}"]}')

    elif tipo_grafico == 'Pronósticos de Modelos':
        for modelo, pronostico in pronosticos.items():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pronostico.index, y=pronostico, mode='lines', name=f'Pronóstico {modelo}', line=dict(dash='dash')))
            fig.update_layout(
                title=f'Pronóstico del modelo {modelo}',
                xaxis_title='Fecha',
                yaxis_title='Valores',
                legend_title='Series',
                hovermode='x unified',
                xaxis_rangeslider_visible=True,
                width=900,  # Ancho del gráfico
                height=600,  # Alto del gráfico
            )
            st.plotly_chart(fig)

            # Crear DataFrame con fechas y pronósticos
            df_pronostico = pd.DataFrame({'Fecha': pronostico.index, 'Pronóstico': pronostico})
            st.subheader(f'DataFrame de Pronóstico {modelo}')
            st.dataframe(df_pronostico)

    elif tipo_grafico == 'Comparación con Datos Reales':
        for modelo, (y_pred, test) in ajustes.items():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Datos reales'))
            fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines', name=f'Ajuste {modelo}', line=dict(dash='dash')))
            fig.update_layout(
                title=f'Comparación con Datos Reales - Modelo {modelo}',
                xaxis_title='Fecha',
                yaxis_title='Valores',
                legend_title='Series',
                hovermode='x unified',
                xaxis_rangeslider_visible=True,
                width=900,  # Ancho del gráfico
                height=600,  # Alto del gráfico
            )
            st.plotly_chart(fig)

# Función para generar pronóstico futuro hasta una fecha específica
def generar_pronostico_hasta_fecha(modelo, series, fecha_final):
    if modelo == 'Holt Winters':
        seasonal_periods = 12  # Ajustar según el modelo Holt Winters utilizado
        alpha = 0.3  # Ajustar según el modelo Holt Winters utilizado
        beta = 0.2  # Ajustar según el modelo Holt Winters utilizado
        gamma = 0.1  # Ajustar según el modelo Holt Winters utilizado
        train = series  # Utilizar todos los datos para entrenamiento
        modelo_hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
        resultado = modelo_hw.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
        
    elif modelo == 'ARIMA':
        order = (5, 1, 0)  # Ajustar según el modelo ARIMA utilizado
        train = series  # Utilizar todos los datos para entrenamiento
        modelo_arima = ARIMA(train, order=order)
        resultado = modelo_arima.fit()

    # Generar fechas de pronóstico hasta la fecha final
    fechas_prediccion = pd.date_range(start=train.index[-1], end=fecha_final)
    pronostico = resultado.predict(start=len(train), end=len(train)+len(fechas_prediccion)-1)

    # Crear DataFrame con fechas y pronósticos hasta la fecha final
    df_pronostico = pd.DataFrame({'Fecha': fechas_prediccion, 'Pronóstico': pronostico})
    return df_pronostico

# Configuración de la aplicación Streamlit
st.title('Aplicación de Modelos de Series Temporales')

# Añadir resumen en la sidebar
st.sidebar.markdown("""
    ## Resumen
    El código crea una aplicación web interactiva donde los usuarios pueden cargar datos de series temporales,
    seleccionar y ajustar modelos de pronóstico (Holt Winters o ARIMA), ver las predicciones y errores del modelo,
    y comparar los resultados en gráficos interactivos. Además, incluye un aviso de copyright en la parte inferior
    de la página.
""")

# Cargar archivo CSV
archivo = st.file_uploader("Subir archivo CSV", type=['csv'])

if archivo is not None:
    datos = cargar_datos(archivo)
    
    if datos is not None:
        st.subheader('Datos cargados:')
        st.write(datos)  # Mostrar todos los registros del DataFrame

        columnas = datos.columns.tolist()
        fecha_col = st.selectbox('Seleccionar columna de fecha', columnas)
        data_col = st.selectbox('Seleccionar columna de datos', columnas)
        
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
            tipo_grafico = st.sidebar.selectbox('Seleccionar tipo de gráfico', ['Ajustes de Modelos', 'Pronósticos de Modelos', 'Comparación con Datos Reales', 'Predicción Futura'])

            ajustes, pronosticos, errores = probar_modelos(series, modelos_seleccionados)

            if tipo_grafico == 'Ajustes de Modelos':
                grafico_interactivo(series, ajustes, pronosticos, errores, tipo_grafico)
            elif tipo_grafico == 'Pronósticos de Modelos':
                grafico_interactivo(series, ajustes, pronosticos, errores, tipo_grafico)
            elif tipo_grafico == 'Comparación con Datos Reales':
                grafico_interactivo(series, ajustes, pronosticos, errores, tipo_grafico)
            elif tipo_grafico == 'Predicción Futura':
                st.subheader('Generar Pronóstico hasta una Fecha Específica')

                modelo_prediccion = st.selectbox('Seleccionar modelo para la predicción', modelos_seleccionados)
                #fecha_final = st.date_input('Seleccionar fecha final para la predicción', min_value=series.index[-1] + timedelta(days=1), max_value=datetime.now())

                fecha_final = st.date_input('Seleccionar fecha final para la predicción',
                            min_value=series.index[-1] + timedelta(days=1),
                            max_value=datetime(2030, 12, 31))


                if st.button('Generar Pronóstico'):
                    if modelo_prediccion and fecha_final:
                        pronostico_df = generar_pronostico_hasta_fecha(modelo_prediccion, series, fecha_final)
                        st.subheader('Pronóstico hasta la Fecha Específica')
                        st.dataframe(pronostico_df)

# Añadir aviso de copyright
st.sidebar.markdown("""
    ---

    &copy; 2024. Desarrollado por jahoperi - Basado en una aplicación de Streamlit para series temporales.
""")