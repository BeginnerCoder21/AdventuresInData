import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

def run():
    df = pd.read_csv('Churn_Modelling.csv')
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    X = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male']]
    y = df['Exited']

    rfc = RandomForestClassifier()
    rfc.fit(X, y)

    st.title('Customer Churn Prediction')
    image_url = "https://img.freepik.com/premium-vector/e-banking-distant-bank-services-experience-finance-control-outline-concept-transactions-withdraw-payment-online-app-illustration-secure-modern-internet-money-management-system_1995-652.jpg?w=2000"
    st.image(image_url,use_column_width=True)
    st.header('Input Parameters')
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.slider('Credit Score', float(df['CreditScore'].min()), float(df['CreditScore'].max()), float(df['CreditScore'].mean()))
        age = st.text_input('Age', float(df['Age'].mean()))
        estimated_salary = st.text_input('Estimated Salary', float())
        has_credit_card = st.checkbox('Has Credit Card?') 
        is_active_member = st.checkbox('Is Active Member?')

    with col2:
        tenure = st.slider('Tenure', float(df['Tenure'].min()), float(df['Tenure'].max()), float(df['Tenure'].mean()))
        balance = st.slider('Balance', float(df['Balance'].min()), float(df['Balance'].max()), float(df['Balance'].mean()))
        num_of_products = st.slider('Number of Products', float(df['NumOfProducts'].min()), float(df['NumOfProducts'].max()), float(df['NumOfProducts'].mean()))

    has_credit_card = 1 if has_credit_card else 0
    is_active_member = 1 if is_active_member else 0

    input_features = pd.DataFrame([[credit_score, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary, 0, 0, 1]],
                                columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male'])

    prediction = rfc.predict(input_features)
    prediction_probability = rfc.predict_proba(input_features)[:, 1]

    if st.button('Predict'):
        st.header('Prediction')
        if prediction[0] == 1:
            st.subheader('The model predicts that the customer will exit.')
        else:
            st.subheader('The model predicts that the customer will not exit.')
        st.write('Prediction Probability:', prediction_probability[0])

