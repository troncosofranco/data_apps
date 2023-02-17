#import library
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Iris Flower Prediction")

#head
st.write("This apps predicts the Iris Flower type using ML classification algorithm")

reference_image = Image.open("references.jpg")
st.image(reference_image, use_column_width=False)


#Sidebar - header
st.sidebar.header("Input selection")

def user_input():

    #input features and ranges
    sepal_length = st.sidebar.slider("Sepal length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal length",1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 0.2)

    #dict with the input features
    data = {'sepal_length': sepal_length,
        "sepal width": sepal_width,
        "petal length": petal_length,
        "petal width": petal_length
    }

    #dict to daframe
    input_features = pd.DataFrame(data, index=[0])
    return input_features

#show user input
df = user_input()
st.subheader('User inputs')
st.write(df)

#Load data for train model
iris = datasets.load_iris()
X = iris.data
y = iris.target

#Build model
model = RandomForestClassifier()
model.fit(X, y)

#make prediction and probability
prediction = model.predict(df)
prediction_probability = model.predict_proba(df)

#Display predictions
st.subheader("Prediction:")
target_name_prediction = iris.target_names[prediction]
st.subheader(target_name_prediction[0])
#st.write(iris.target_names[0]) #labels and coded values of target feature

flower_type = str(target_name_prediction[0])
if flower_type == "setosa":
    image_type = Image.open("iris_setosa.jpg")
elif flower_type == "virginica": 
    image_type = Image.open("iris_virginica.jpg")
else:
    image_type = Image.open("iris_versicolor.jpg")
    

st.image(image_type, use_column_width=False)



st.subheader("Prediction probability:")
results = {'Type':['Setosa', 'Versicolor', 'Virginica'],
        'Probability':[prediction_probability[0,0],prediction_probability[0,1], prediction_probability[0,2]]}
  
# Create DataFrame
df_results = pd.DataFrame(results)


fig, ax = plt.subplots(figsize=(5,3))
sns.barplot(x=df_results.Type, y=df_results.Probability, color='blue', ax=ax, label="Probability plot")
ax.set_xlabel("Type")
ax.set_ylabel("Probability")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


st.markdown("""
Credits:
* **Images:** [Unplash] (https://unsplash.com/es/s/fotos/iris-flower) and [google] (https://www.google.com)
* **Code contribution:** [Dataprofessor](https://github.com/dataprofessor)
""")
