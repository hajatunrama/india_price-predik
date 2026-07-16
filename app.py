import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model
@st.cache_resource
def load_model():
    model = joblib.load('best_laptop_price_model.pkl')
    return model

model = load_model()

st.title("💻 Prediksi Harga Laptop (Machine Learning)")
st.write("Aplikasi ini memprediksi harga laptop berdasarkan spesifikasi menggunakan algoritma terbaik dari penelitian.")

# Buat Form Input berdasarkan fitur dataset
st.sidebar.header("Input Spesifikasi Laptop")

def user_input_features():
    Brand = st.sidebar.selectbox('Brand', ['Apple', 'Lenovo', 'ASUS', 'Dell', 'Samsung', 'Huawei', 'Xiaomi', 'LG', 'Acer', 'MSI', 'Gigabyte', 'HP'])
    Model = st.sidebar.selectbox('Model', ['Inspiron', 'Victus', 'ZenBook', 'MacBook Pro', 'Legion', 'ROG', 'TUF', 'ThinkPad', 'IdeaPad', 'MacBook Air'])
    Processor_Brand = st.sidebar.selectbox('Processor Brand', ['Intel', 'AMD', 'Apple'])
    Processor = st.sidebar.selectbox('Processor', ['Core i5', 'Core i7', 'Core i3', 'Ryzen 5', 'Ryzen 7', 'M3'])
    Generation = st.sidebar.slider('Generation', min_value=11, max_value=15, value=13)
    RAM_GB = st.sidebar.selectbox('RAM (GB)', [8, 16, 32])
    RAM_Type = st.sidebar.selectbox('RAM Type', ['DDR4', 'DDR5'])
    Storage_GB = st.sidebar.selectbox('Storage (GB)', [256, 512, 1024])
    Storage_Type = st.sidebar.selectbox('Storage Type', ['SSD', 'HDD'])
    Graphics = st.sidebar.selectbox('Graphics', ['Integrated', 'RTX 4050', 'RTX 4060', 'RTX 3050'])
    Graphics_Type = st.sidebar.selectbox('Graphics Type', ['Integrated', 'Dedicated'])
    Display_Size_Inches = st.sidebar.slider('Display Size (Inches)', 13.0, 16.0, 15.6)
    Resolution = st.sidebar.selectbox('Resolution', ['1920x1080', '2560x1600'])
    Refresh_Rate_Hz = st.sidebar.selectbox('Refresh Rate (Hz)', [60, 120, 144])
    Touchscreen = st.sidebar.selectbox('Touchscreen', ['No', 'Yes'])
    Battery_Wh = st.sidebar.slider('Battery (Wh)', 42, 90, 70)
    Weight_kg = st.sidebar.slider('Weight (kg)', 1.2, 2.6, 1.8)
    Operating_System = st.sidebar.selectbox('OS', ['Windows 11', 'Ubuntu', 'macOS'])
    Warranty_Years = st.sidebar.slider('Warranty (Years)', 1, 3, 1)
    Color = st.sidebar.selectbox('Color', ['Blue', 'Gray', 'Silver', 'Black'])
    Release_Year = st.sidebar.slider('Release Year', 2023, 2026, 2024)

    # Buat DataFrame
    data = {
        'Brand': Brand, 'Model': Model, 'Processor_Brand': Processor_Brand,
        'Processor': Processor, 'Generation': Generation, 'RAM_GB': RAM_GB,
        'RAM_Type': RAM_Type, 'Storage_GB': Storage_GB, 'Storage_Type': Storage_Type,
        'Graphics': Graphics, 'Graphics_Type': Graphics_Type, 'Display_Size_Inches': Display_Size_Inches,
        'Resolution': Resolution, 'Refresh_Rate_Hz': Refresh_Rate_Hz, 'Touchscreen': Touchscreen,
        'Battery_Wh': Battery_Wh, 'Weight_kg': Weight_kg, 'Operating_System': Operating_System,
        'Warranty_Years': Warranty_Years, 'Color': Color, 'Release_Year': Release_Year
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("Spesifikasi yang Dipilih")
st.write(input_df)

# Tombol Prediksi
if st.button("Prediksi Harga"):
    prediction = model.predict(input_df)
    st.subheader("Hasil Prediksi")
    # Format harga ke format Rupee India (INR)
    st.success(f"Estimasi Harga Laptop: ₹ {prediction[0]:,.2f}")
    st.caption("Catatan: Harga diprediksi dalam satuan INR (Indian Rupee) sesuai dataset.")
