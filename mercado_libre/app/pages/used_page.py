import streamlit as st
import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np


def used_page():
    
    #header
    st.header("Car Price: Prediction")
    st.write("""
        This app predicts the price of used cars based on features 🚗!
        """)
    st.write('------------')

    #Load clean dataframe to obtain the categorical values
    df = pd.read_csv('data/used_data_app.csv')
    
    #Input data
    company = st.sidebar.selectbox('Company:', df.company.unique())
    model = st.sidebar.selectbox('Model:', df.model.unique())
    year = st.sidebar.slider('Year:', float(df.year.min()), float(df.year.max()), float(df.year.mean()))
    fuel = st.sidebar.selectbox('Fuel:', df.fuel.unique())
    doors = st.sidebar.slider('Doors:', float(df.doors.min()), float(df.doors.max()), float(df.doors.mean()))
    transmisions = st.sidebar.selectbox('Transmision:', df.transmisions.unique())
    engine = st.sidebar.slider('Engine Size:', float(df.motor.min()), float(df.motor.max()), float(df.motor.mean()))
    bodywork = st.sidebar.selectbox('Bodywork', df.bodywork.unique())
    km = st.sidebar.slider('km:', float(df.km.min()), float(df.km.max()), float(df.km.mean()))
    
    #Load pipeline
    pipeline = joblib.load('./models/pipe_used.joblib')
    
    
    prediction = pipeline.predict(pd.DataFrame(columns=['company','model','year','fuel', 'doors','transmisions','motor','bodywork','km'],
            data=np.array([company,model,year,fuel,doors, transmisions,engine,bodywork,km]).reshape(1,9)))[0]
    
    st.subheader('Result')
    st.success(f"The prediction is: $ {str(round(prediction, 2))}")

    st.balloons()

    
    return