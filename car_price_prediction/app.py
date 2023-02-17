#1. Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import shap


st.header("Car Price: Analysis & Prediction")
st.write("""
This app predicts the price based on car features 🚗!
""")
st.write('------------')




#2. Import dataset 
# source:https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv
df = pd.read_csv('car_price_dataset.csv')

#3. Data overview
#st.write(df.isnull().sum()) #null values
#st.write(df.describe())
#st.write(df.info())

#4. Heatmap of correlation matrix
st.subheader('Heatmap of matrix correlation')
plt.figure(figsize=(20, 15))
df_corr = df.corr()
mask = np.triu(np.ones_like(df.corr()))
sns.heatmap(df_corr, cmap="coolwarm", annot=True, mask=mask)

plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


#5. Prediction model
from sklearn.model_selection import train_test_split

label = "price" #target feature
#Correlated variables

df = df[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]


X = np.array(df.drop([label], 1))
y = np.array(df[label])



#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)



#6. Prediction

predictions = model.predict(X_test)
from sklearn.metrics import mean_absolute_error

st.subheader('Price Prediction')
st.write('The Mean Absolute Error of the model is ', str(model.score(X_test, predictions))) 

st.sidebar.header('Prediction Parameters Selection')

symboling = st.sidebar.slider('Symboling', float(df.symboling.min()), float(df.symboling.max()), float(df.symboling.mean()))
wheelbase = st.sidebar.slider('Wheelbase', float(df.wheelbase.min()), float(df.wheelbase.max()), float(df.wheelbase.mean()))
carlength = st.sidebar.slider('Car Length', float(df.carlength.min()), float(df.carlength.max()), float(df.carlength.mean()))
carwidth = st.sidebar.slider('Car Width', float(df.carwidth.min()), float(df.carwidth.max()), float(df.carwidth.mean()))
carheight = st.sidebar.slider('Car Height', float(df.carheight.min()), float(df.carheight.max()), float(df.carheight.mean()))
curbweight = st.sidebar.slider('Curb Weight', float(df.curbweight.min()), float(df.curbweight.max()), float(df.curbweight.mean()))
enginesize = st.sidebar.slider('Engine Size', float(df.enginesize.min()), float(df.enginesize.max()), float(df.enginesize.mean()))
boreratio = st.sidebar.slider('Bore Ratio', float(df.boreratio.min()), float(df.boreratio.max()), float(df.boreratio.mean()))
stroke = st.sidebar.slider('Stroke', float(df.stroke.min()), float(df.stroke.max()), float(df.stroke.mean()))
compressionratio = st.sidebar.slider('Compression Ratio', float(df.compressionratio.min()), float(df.compressionratio.max()), float(df.compressionratio.mean()))
horsepower = st.sidebar.slider('Horsepower', float(df.horsepower.min()), float(df.horsepower.max()), float(df.horsepower.mean()))
peakrpm = st.sidebar.slider('Peakrpm', float(df.peakrpm.min()), float(df.peakrpm.max()), float(df.peakrpm.mean()))
citympg = st.sidebar.slider('City mpg', float(df.citympg.min()), float(df.citympg.max()), float(df.citympg.mean()))
highwaympg = st.sidebar.slider('Highway mpg', float(df.highwaympg.min()), float(df.highwaympg.max()), float(df.highwaympg.mean()))

#input data to a dictonary    
data = {'symboling':symboling, 
        "wheelbase": wheelbase,
        'carlength': carlength,
        "carwidth": carwidth,
        "carheight": carheight,
        "curbweight": curbweight,
        "enginesize": enginesize, 
        "boreratio": boreratio, 
        "stroke": stroke,
        "compressionratio":compressionratio,
        "horsepower": horsepower,
        "peakrpm": peakrpm, 
        "citympg": citympg, 
        "highwaympg": highwaympg
        }
        

#data input to dataframe   
df_input = pd.DataFrame(data, index=[0])
st.write('Input parameters')
st.write(df_input)

prediction =model.predict(df_input)
st.subheader('Result')
st.write(f"The prediction is: {str(round(prediction[0], 2))}$")

#Shap graphics
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

X_shape = df.drop([label], 1)
st.subheader('Feature Contribution')
shap.summary_plot(shap_values, X_shape)
st.pyplot(bbox_inches='tight')


    

