import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb  
from sklearn.model_selection import train_test_split

def clean_data(df):
        df.drop(columns=['Unnamed: 0','Status'], inplace=True)
        df['Furnished_status'].fillna(df['Furnished_status'].mode()[0], inplace=True)
        df.isnull().sum()
        df['Bedrooms'] = df['Bedrooms'].astype('int')
        df['Bathrooms'] = df['Bathrooms'].astype('int')
        df.drop(columns=['latitude', 'longitude', 'Lift', 'parking','desc','Lift','Balcony'], inplace=True)
        location_counts = df['Address'].value_counts()
        locations_to_replace = location_counts[location_counts <= 10].index
        df.loc[df['Address'].isin(locations_to_replace), 'Address'] = 'Other'
        df['Address'].value_counts()
        df['Location'] = df['Address'].str.cat(df['Landmarks'], sep=', ')
        df = df.dropna()
        df = df.drop(['Address', 'Landmarks'], axis=1)
        df.head()
        df.drop(columns='Price_sqft', inplace=True)
        cleaned_df=df[['price','area', 'Bedrooms', 'Bathrooms','Location','neworold','Furnished_status','type_of_building']]
        cleaned_df.head()

        return cleaned_df


def run():
    st.title("House Price Prediction App")

    file_path = "Delhi_v2.csv"
    df = pd.read_csv(file_path)
    df = clean_data(df)

    col1, col2 = st.columns(2)
    location = st.selectbox('Location', df['Location'].unique())
    with col1:
        area = st.slider("Area (sqft)", min_value=df['area'].min(), max_value=df['area'].max())
        Bedrooms = st.number_input("Bedrooms", min_value=1)
        neworold = st.selectbox('Property Type', df['neworold'].unique())
    with col2:
        Bathrooms = st.number_input("Bathrooms", min_value=1)
        Furnished_status = st.selectbox('Furnished', df['Furnished_status'].unique())
        type_of_building = st.selectbox('Building Type', df['type_of_building'].unique())

    if st.button("Predict"):
        features = np.array([area, Bedrooms, Bathrooms, location, neworold, Furnished_status, type_of_building])
        X = df.drop(columns='price')
        y = np.log(df['price'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        column_trans = make_column_transformer(
            (OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first'),
             [3,4,5,6]),
            remainder='passthrough'
        )
        scaler = StandardScaler()
        xgboost_regressor = xgb.XGBRegressor()
        pipe = make_pipeline(column_trans, scaler, xgboost_regressor)
        pipe.fit(X_train, y_train)
        query = features.reshape(1, -1)
        predictions = pipe.predict(query)
        predicted_price = int(np.exp(predictions)[0])
        st.header("Predicted Price")
        st.header(predicted_price)

if __name__ == '__main__':
    run()
