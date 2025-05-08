# Electricity Saving Assistant with Extended Features
import streamlit as st
import pandas as pd
import joblib
import json
import random
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="Electricity Saving Assistant", page_icon="⚡", layout="centered")

# Define base path for relative file access
BASE_DIR = os.path.dirname(__file__)

# Load model and encoders safely
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
mlb = joblib.load(os.path.join(BASE_DIR, "appliance_encoder.pkl"))

# Load tips database
with open(os.path.join(BASE_DIR, "tips.json"), "r") as f:
    tips_db = json.load(f)

# App title
st.title("Electricity Saving Assistant ⚡")

# Monthly electricity input
units = st.slider("Select your monthly electricity consumption (kWh):", 0, 2000, 200, 100)

# Appliance selection
appliances_list = list(tips_db.keys())
selected_appliances = st.multiselect("Select the appliances you use:", appliances_list)

# Daily usage inputs
usage_times = {}
if selected_appliances:
    st.subheader("🕒 Daily Usage Time per Appliance")
    for appliance in selected_appliances:
        usage_times[appliance] = st.slider(f"{appliance} (hrs/day)", 0.0, 24.0, 1.0, 0.5)

# Predict and display results
if st.button("Predict Energy Savings"):
    if not selected_appliances:
        st.warning("Please select at least one appliance.")
    else:
        # Encode appliances
        appliance_encoded = mlb.transform([selected_appliances])
        prediction = model.predict(appliance_encoded)[0]

        st.success(f"Estimated Energy Saving Potential: {prediction:.2f} kWh/month")

        # Show random tips
        st.subheader("💡 Energy Saving Tips")
        for appliance in selected_appliances:
            tips = tips_db.get(appliance, [])
            if tips:
                tip = random.choice(tips)
                st.markdown(f"**{appliance}**: {tip}")

# Optional: Visualize usage time
if usage_times:
    st.subheader("📊 Appliance Usage Distribution")
    fig, ax = plt.subplots()
    ax.pie(usage_times.values(), labels=usage_times.keys(), autopct="%1.1f%%")
    st.pyplot(fig)
