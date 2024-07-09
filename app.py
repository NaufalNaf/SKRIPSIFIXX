import joblib
import streamlit as st

# Load save model dan scaler
model = joblib.load('klasifikasi_obesitas_svm.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Klasifikasi BMI App')

# User input
gender = st.radio('Gender', ['Male', 'Female'])
height = st.number_input('Height (cm)', min_value=0.0)
weight = st.number_input('Weight (kg)', min_value=0.0)

gender_num = 0 if gender == 'Female' else 1
# preprocessing 
input_data = [[gender_num, height, weight]]
input_scaled = scaler.transform(input_data)

# Prediction
if st.button('Prediksi'):
    prediction = model.predict(input_scaled)[0]
    index_labels = {
        0: 'Sangat Kurus',
        1: 'Kurus',
        2: 'Normal',
        3: 'Kelebihan Berat',
        4: 'Obesitas',
        5: 'Obesitas Extreme'
    }
    st.write(f'Kategori BMI Anda: {index_labels[prediction]}')

