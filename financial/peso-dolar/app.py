#Import modules 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly

#Head
st.title('Dólar futuro (U$d)')
image = Image.open("logo_dólar.jpg")
st.image(image, use_column_width=True)
#Data updated to 21/7/2022 from https://www.ambito.com/contenidos/


#Define stocks
stocks = ('Dólar Blue','Dólar Oficial', "CCL", "Dólar MEP")
selected_stock = st.sidebar.selectbox('Dataset prediction', stocks)
buy_sell = ['Compra', "Venta"]
operation = st.sidebar.radio("Operación", buy_sell)
start = st.sidebar.date_input("Desde",)
end = st.sidebar.date_input("Hasta",)

#Define prediction period
n_months = st.slider('Semanas de predicción', 1, 12)
period = n_months * 7


#Load data
if selected_stock == 'Dólar Oficial':
  df = pd.read_excel("dolar_oficial.xlsx", parse_dates=True)
elif selected_stock == 'CCL':
  df = pd.read_excel("CCL.xlsx", parse_dates=True)
elif selected_stock == 'Dólar MEP':
  df = pd.read_excel("dolar_MEP.xlsx", parse_dates=True)
else:
  df = pd.read_excel("dolar_blue.xlsx", parse_dates=True)  
  

#Format data time
#df.index = df['Date']
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.date
#df['Date'] = pd.to_datetime(df['Date'])

#df['Date'] = df['Date'].apply(lambda x: x.date()) #remove hours from date
#df['Date']= df['Date'],format='%Y-%m-%d', yearfirst= True).dt.date #remove hours from date

#Filter data
df_filter = df[df["Date"] >= start] 
df_filter = df_filter[df_filter["Date"] <= end]

#Plot load
def price_plot():
  plt.fill_between(df_filter['Date'], df_filter[operation], color='skyblue', alpha=0.3)
  plt.plot(df_filter['Date'], df_filter[operation], color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(selected_stock, fontweight='bold')
  plt.xlabel('Fecha', fontweight='bold')
  plt.ylabel('Precio ($)', fontweight='bold')
  st.set_option('deprecation.showPyplotGlobalUse', False)
  return st.pyplot()
 
#price_plot()

#Density function
def density_plot():
  st.write('Densidad de datos 📊...Cuantas veces se repite el valor en el intervalo de tiempo')
  fig, ax = plt.subplots(figsize=(15, 7))
  sns.distplot(df_filter[operation], ax=ax)
  st.set_option('deprecation.showPyplotGlobalUse', False)
  return st.pyplot(fig)

#Boxplot function
def box_plot():
  df_filter['Date'] = pd.to_datetime(df['Date'])
  df_filter['year'] = df_filter['Date'].dt.year #daily data to yearly data
  
  ax = sns.boxplot(x=df_filter['year'], y=df_filter[operation], data=df_filter) 
  ax = sns.stripplot(x=df_filter['year'], y=df_filter[operation], data=df_filter, color="orange", jitter=0.2, size=2.5)
  
  plt.title(selected_stock, fontweight='bold')
  plt.xlabel('Año', fontweight='bold')
  plt.ylabel('Precio', fontweight='bold')  
  st.set_option('deprecation.showPyplotGlobalUse', False)
  return st.pyplot()


#Forecast function
def prediction():
  m = Prophet()
  df_train = df_filter.rename(columns={'Date': 'ds', operation: 'y'})
  st.write('Prediciendo 💲⬆️...')
  m.fit(df_train)
  future = m.make_future_dataframe(periods=period)
  forecast = m.predict(future)
  forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
  fig1 = plot_plotly(m, forecast)
    
  return st.plotly_chart(fig1)


#Buttons
#Plot button
if st.button('Gráficar'):
    st.header('Gráfica')
    price_plot()

#Density button
if st.button('Densidad'):
    st.header('Densidad')
    density_plot()

#Boxplot button
if st.button('Boxplot Anual'):
    st.header('Boxplot Anual')
    box_plot()

#Prediction button
if st.button('Predicción'):
    st.header('Dólar vs pesos')
    prediction()
