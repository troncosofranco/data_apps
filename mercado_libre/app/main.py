import streamlit as st
from streamlit_option_menu import option_menu
from pages.new_page import new_page
from pages.used_page import used_page
from PIL import Image

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 300px;
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add the image header to the sidebar
#image = Image.open("car_price_logo.jpg")
#st.image(image, use_column_width=False)


selected = option_menu(None, ["New Car", "Used Car"], 
    icons=['gear', 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
)

if selected == 'New Car':
    new_page()
if selected == 'Used Car':
    used_page()
