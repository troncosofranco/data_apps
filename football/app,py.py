#libraries
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

st.title('NFL Football: Rushing Stats Analysis')

#head image
image = Image.open("C:\\Users\\tronc\\Desktop\\Python\\data_science\\apps\\EDA_football\\NFL.jpg")
st.image(image, use_column_width=True)



st.sidebar.header('Selecting options')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(2000,2022))))

# Web scraping of NFL player stats function
# https://www.pro-football-reference.com/years/2019/rushing.htm
@st.cache
def load_data(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/rushing.htm"
    html = pd.read_html(url, header = 1)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0) #fill empty values with 0
    playerstats = raw.drop(['Rk'], axis=1) #Remove header
    return playerstats
playerstats = load_data(selected_year)

# Sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['RB','QB','WR','FB','TE'] #option position vector
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data according to user inputs
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Player of Selected Teams')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download dataset
#https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-file-streamlit
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="NFL_dataset.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap matrix
if st.button('Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

#Boxplot
if st.button('Boxplot'):
    st.header('Boxplot')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    df_plot = df.drop(columns=['Tm', 'Pos'])
    
    df_plot.plot(kind='box')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

st.subheader("Select variables for scatterplot")
vector_features = ['Age', 'G', 'GS', 'Att', 'Yds', 'TD', '1D', 'Lng', 'Y/A', 'Y/G', 'Fmb']
selected_x = st.selectbox('x variable', list(vector_features))
selected_y = st.selectbox('y variable', list(vector_features))



#scatterplot
if st.button('Scatterplot'):
    st.header('Scatterplot')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    sns.jointplot(x=df[selected_x], y=df[selected_y], kind='hex',marginal_kws=dict(bins=10, fill=True))
    st.pyplot()

st.write("---")
st.markdown("""
References:
* **Tm:** Team 
* **Age:** Player's ageat the end of the season 
* **Pos:** Position. Capital letter: Primary starter, Lower-case: Part-time starter, Upper-case: Full-time starter  
* **G:** Games played
* **GS:** Games started as an offensive or defensive player
* **Att:** Rushing attempts
* **Yds:** Rushing yards
* **TD:** Rushing touchdowns
* **1D:** Rushing 1st downs
* **Lng:** Rushing longest rush
* **Y/A:** Rushing yards per attempt
* **Y/G:** Rushing yards per game
* **Fmb:** Rushing fumble returns
""")



st.write("---")
st.markdown("""
Credits:
* **Image:** [Freepik] (https://www.freepik.es/fotos-premium/silueta-jugador-futbol-americano-llamas_10938203.htm#query=NFL%20on%20fire&position=23&from_view=search)
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/).
* **Code contribution:** [Dataprofessor](https://github.com/dataprofessor/streamlit_freecodecamp/tree/main/app_4_eda_football)
""")

