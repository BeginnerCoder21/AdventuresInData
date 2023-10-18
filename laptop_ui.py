import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from laptop_code import preprocess, predict

def run():
    st.title("Laptop Price Prediction")
    image= Image.open("image.jpg")
    st.image(image, use_column_width=True)

    file_path = "laptop_data.csv" 
    df = pd.read_csv(file_path)
    df=preprocess(df)
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox('Brand', df['company'].unique())
        ram= st.selectbox('RAM (in GB)', [2,4,6,8,16,24,32,64])
        col3, col4 = st.columns(2)
        with col3:
            touchscreen= st.radio('Touchscreen?', ('Yes', 'No'))
        with col4:
            ips= st.radio('IPS Diaplay?', ('Yes', 'No'))
        weight = st.slider("Select weight",min_value=0.0, max_value=5.0)
        resolution=st.selectbox('Screen Resolution',['1920 X 1080','1366 x 768', '1600 X 900','3840 X 2160', '3200 X 1800', '2560 X 1600',
                                                '2560 X 1440' , '2304 X 1440'])
        gpu = st.selectbox('GPU', df['gpu_type'].unique())

    with col2:
        type = st.selectbox('Type', df['typename'].unique())
        hdd= st.selectbox('HDD (in GB)',[0,128,256,512, 1024, 2048])
        ssd= st.selectbox('SSD (in GB)',[0,8, 128,256,512, 1024])
        selected_size= st.number_input("Enter a size")
        cpu=st.selectbox('Brand', df['cpu_brand'].unique())
        os=st.selectbox('Operating System', df['os'].unique())
    st.markdown(
        """
        <style>
        .custom-text-color {
            color: #FFCF2A; 
            font-size: 30px;
        }
        .head-color{
        color: #FFCF2A;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button('Predict Price'):
        ppi=None
        if touchscreen=="Yes":
            touchscreen=1
        else:
            touchscreen=0
        
        if ips=="Yes":
            ips=1
        else:
            ips=0
        X_res, Y_res = map(int, resolution.split(' X '))

        ppi= ((X_res**2)+(Y_res**2))**0.5/selected_size
        features = np.array([brand,type,ram,weight, touchscreen, ips, ppi, cpu,hdd, ssd, gpu, os])  
        predicted_price = predict(df, features)

        st.subheader(f"Predicted Price of current specs is:")
        st.markdown(f'<h1 class="head-color"> Rs. {predicted_price} </h1>', unsafe_allow_html=True)
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.text(f"Price in USD")
            st.markdown(f'<h2 class="custom-text-color">${round(predicted_price*0.012,3)}</h2>', unsafe_allow_html=True)
        with col6:
            st.text(f"Price in CNY")
            st.markdown(f'<h2 class="custom-text-color">¥{round(predicted_price*0.088,3)}</h2>', unsafe_allow_html=True)
        with col7:
            st.text(f"Price in JPY")
            st.markdown(f'<h2 class="custom-text-color">円{round(predicted_price*1.80,3)}</h2>', unsafe_allow_html=True)
        with col8:
            st.text(f"Price in EUR")
            st.markdown(f'<h2 class="custom-text-color">€{round(predicted_price*0.011, 3)}</h2>', unsafe_allow_html=True)