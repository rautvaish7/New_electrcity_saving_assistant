import streamlit as st
import pandas as pd
import joblib
import json
import random
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(page_title="Electricity Saving Assistant", page_icon="⚡", layout="centered")

# Define base directory
BASE_DIR = os.path.dirname(__file__)

# Load the trained model and encoders
model_path = os.path.join(BASE_DIR, "model.pkl")
encoder_path = os.path.join(BASE_DIR, "appliance_encoder.pkl")
tips_path = os.path.join(BASE_DIR, "tips.json")

try:
    model = joblib.load(model_path)
    mlb = joblib.load(encoder_path)
    with open(tips_path, "r") as f:
        tips_db = json.load(f)
except Exception as e:
    st.error(f"Error loading model or files: {str(e)}")
    st.stop()

# App title
st.title("Electricity Saving Assistant ⚡")

# Electricity usage input
units = st.slider("Select your monthly electricity consumption (kWh):", 0, 2000, 200, 100)

# Appliance selection
appliances_list = list(tips_db.keys())
selected_appliances = st.multiselect("Select the appliances you use:", appliances_list)

# Daily usage time
usage_times = {}
if selected_appliances:
    st.subheader("🕒 Daily Usage Time per Appliance")
    for appliance in selected_appliances:
        usage_times[appliance] = st.slider(f"{appliance} (hrs/day)", 0.0, 24.0, 1.0, 0.5)

# Prediction
if st.button("Predict Energy Savings"):
    if not selected_appliances:
        st.warning("Please select at least one appliance.")
    else:
        try:
            # Encode appliances
            appliance_encoded = mlb.transform([selected_appliances])
            st.write("Encoded shape:", appliance_encoded.shape)
            st.write("Model type:", type(model))

            # Predict
            prediction = model.predict(appliance_encoded)[0]
            st.success(f"Estimated Energy Saving Potential: {prediction:.2f} kWh/month")

            # Show random tips
            st.subheader("💡 Energy Saving Tips")
            for appliance in selected_appliances:
                tips = tips_db.get(appliance, [])
                if tips:
                    tip = random.choice(tips)
                    st.markdown(f"**{appliance}**: {tip}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Visualization of usage
if usage_times:
    st.subheader("📊 Appliance Usage Distribution")
    fig, ax = plt.subplots()
    ax.pie(usage_times.values(), labels=usage_times.keys(), autopct="%1.1f%%")
    st.pyplot(fig)
