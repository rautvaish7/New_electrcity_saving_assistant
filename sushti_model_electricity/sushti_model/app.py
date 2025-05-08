import streamlit as st
import pandas as pd
import joblib
import json
import random
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="Electricity Saving Assistant", page_icon="⚡", layout="centered")

# Define base path
BASE_DIR = os.path.dirname(__file__)

# Load model and files safely
try:
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))  # NearestNeighbors
    mlb = joblib.load(os.path.join(BASE_DIR, "appliance_encoder.pkl"))
    with open(os.path.join(BASE_DIR, "tips.json"), "r") as f:
        tips_db = json.load(f)
except Exception as e:
    st.error(f"Error loading model or files: {str(e)}")
    st.stop()

# UI
st.title("Electricity Saving Assistant ⚡")
units = st.slider("Select your monthly electricity consumption (kWh):", 0, 2000, 200, 100)

appliances_list = list(tips_db.keys())
selected_appliances = st.multiselect("Select the appliances you use:", appliances_list)

usage_times = {}
if selected_appliances:
    st.subheader("🕒 Daily Usage Time per Appliance")
    for appliance in selected_appliances:
        usage_times[appliance] = st.slider(f"{appliance} (hrs/day)", 0.0, 24.0, 1.0, 0.5)

# Predict using NearestNeighbors
if st.button("Predict Energy Savings"):
    if not selected_appliances:
        st.warning("Please select at least one appliance.")
    else:
        try:
            # Transform input
            appliance_encoded = mlb.transform([selected_appliances])

            # Find k-nearest neighbors
            distances, indices = model.kneighbors(appliance_encoded, n_neighbors=3)
            average_distance = distances[0].mean()

            # Simulate energy savings score (mock logic)
            max_saving = 100  # Max 100 kWh saving
            saving_score = max(0, max_saving - average_distance * 100)  # Lower distance = higher saving

            st.success(f"Estimated Energy Saving Potential: {saving_score:.2f} kWh/month")

            # Show tips
            st.subheader("💡 Energy Saving Tips")
            for appliance in selected_appliances:
                tips = tips_db.get(appliance, [])
                if tips:
                    tip = random.choice(tips)
                    st.markdown(f"**{appliance}**: {tip}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Pie chart
if usage_times:
    st.subheader("📊 Appliance Usage Distribution")
    fig, ax = plt.subplots()
    ax.pie(usage_times.values(), labels=usage_times.keys(), autopct="%1.1f%%")
    st.pyplot(fig)
