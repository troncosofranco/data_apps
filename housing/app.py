import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import shap
from PIL import Image
import seaborn as sns


# Head
st.header("Housing Price Prediction App")
st.write("""
# Boston House Price Prediction App
This app predicts the Boston house price!
""")
st.write('------------')

housing_logo = Image.open("housing_logo.png")
st.image(housing_logo, use_column_width=True)

# Load the Boston House Price Dataset
raw_data = datasets.load_boston()
data_plot = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)

#Also boston housing train and test data can be employed
#st.write(raw_data)

#Split data
X = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
y = pd.DataFrame(raw_data.target, columns=["MEDV"])


st.markdown("""Feature references:
* **CRIM:** per capita crime rate by town
* **ZN:** proportion of residential land zoned for lots over 25,000 sq.ft.
* **INDUS:** proportion of non-retail business acres per town.
* **CHAS:** Charles River dummy variable (1 if tract bounds river; 0 otherwise)
* **NOX:** nitric oxides concentration (parts per 10 million)
* **RM:** average number of rooms per dwelling
* **AGE:** proportion of owner-occupied units built prior to 1940
* **DIS:** weighted distances to five Boston employment centres
* **RAD:** index of accessibility to radial highways
* **TAX:** full-value property-tax rate per $10,000
* **PTRATIO:** pupil-teacher ratio by town
* **B:** 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
* **LSTAT:** lower status of the population (%)
* **MEDV:** Median value of owner-occupied homes in $1000's  
""")


# Header Sidebar
st.sidebar.header('Parameters Selection')

#float for the slider
CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))

#input data to a dictonary    
data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}

#data input to dataframe   
df_input = pd.DataFrame(data, index=[0])

#Plotting section

st.subheader("Select features to plot")
features_vector = ['CRIM','ZN','INDUS','CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']            

selected_x = st.selectbox('Feature x', features_vector)
selected_y = st.selectbox('Feature y', features_vector)

########################################################
#Plot functions
########################################################

#violin plot
def violin_plot():
    sns.violinplot(x=data_plot[str(selected_x)], y=data_plot[str(selected_y)], data=data_plot)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    return st.pyplot()

#distribution plot
def distribution():
    sns.FacetGrid(data_plot, height=6,).map(sns.kdeplot, str(selected_y),shade=True).add_legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    return st.pyplot()

#scatter plot
def scatter_plot():
    sns.scatterplot(data=data_plot, x=data_plot[str(selected_x)], y=data_plot[str(selected_y)])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    return st.pyplot()

#Buttons
#Plot button
if st.button('Violin plot'):
    st.header('violin plot')
    violin_plot()

if st.button('Distribution'):
    st.header('Property distribution')
    distribution()

if st.button('Scatter plot'):
    st.header('Scatter plot')
    scatter_plot()

st.write("---")

#Prediction section
# Print specified input parameters
st.header('MEDV prediction')
st.subheader('Input parameters')
st.write(df_input)


# Build Regression Model
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(max_depth=15)
model.fit(X,y)

prediction =model.predict(df_input)


st.subheader('Result')
st.write(f"The prediction is: {str(round(prediction[0], 2))}K$")


#Use of shap values to explain the model at:
#https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

def plot_shap_values():
    st.subheader('Feature Contribution')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    return st.pyplot(bbox_inches='tight')
    
def plot_shap_bar():
    plt.title('Feature contribution (SHAP values-Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    return st.pyplot(bbox_inches='tight')

if st.button('Shap values'):
    plot_shap_values()

if st.button('Shap bar'):
    plot_shap_bar()

st.markdown("""
Credits:
* **Images:** [Freepik] (https://www.freepik.com)
* **Crecits:** [Dataprofessor] (https://github.com/dataprofessor/streamlit_freecodecamp/tree/main/app_9_regression_boston_housing), 
[Sainithish] (https://www.kaggle.com/code/sainithish1212/top10-rank-with-very-good-score),
[Sharp Tutorial] (https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137) by Vinicius Trevisan
""")