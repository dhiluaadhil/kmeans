import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Stock Market Segmenter", page_icon="📈")
st.title("📈 Stock Market Clustering (K-Means)")

# Load Assets
model_path = 'mbg_kmeans_model.pkl'
scaler_path = 'mbg_scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    st.write("Enter the stock metrics for a specific day to see its cluster segment.")

    # Input fields based on CSV columns
    col1, col2 = st.columns(2)
    with col1:
        open_price = st.number_input("Open Price", value=40.0)
        high_price = st.number_input("High Price", value=41.5)
        low_price = st.number_input("Low Price", value=39.5)
    
    with col2:
        close_price = st.number_input("Close Price", value=41.0)
        volume = st.number_input("Volume", value=3000000)

    if st.button("Identify Market Segment"):
        # Features must be in order: ['Open', 'High', 'Low', 'Close', 'Volume']
        raw_data = np.array([[open_price, high_price, low_price, close_price, volume]])
        
        # Scale and Predict
        scaled_data = scaler.transform(raw_data)
        cluster = model.predict(scaled_data)
        
        st.success(f"### This trading day belongs to: **Group {cluster[0]}**")
        st.info("Different groups often represent different market behaviors (e.g., bull vs bear days).")
else:
    st.error("Please upload 'mbg_kmeans_model.pkl' and 'mbg_scaler.pkl' to your GitHub repo.")