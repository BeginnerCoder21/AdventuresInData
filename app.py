import streamlit as st
import bank
import laptop_ui
import house_ui

def run_bank_customer_churn():
    bank.run()

def run_laptop_price_prediction():
    laptop_ui.run()

def run_house_price_prediction():
    house_ui.run()

selected_project = st.sidebar.radio("Select Project", ["Home", "Laptop Price Prediction","Bank Customer Churn","House Price Prediction"])

if selected_project == "Home":
    st.subheader("Hello 🙌")
    st.title("A Showcase of Data Science Projects")
    st.subheader("By Beginner Coder21")
    st.info('If you a LAPTOP user, please select a project on the left sidebar to view the project.', icon="💻")
    st.success('If you a MOBILE user, please click on \'>\' on top left corner and select a project on the left sidebar to view the project.', icon="📱")
    
if selected_project == "Bank Customer Churn":
    run_bank_customer_churn()
elif selected_project == "Laptop Price Prediction":
    run_laptop_price_prediction()
elif selected_project == "House Price Prediction":
    run_house_price_prediction()

