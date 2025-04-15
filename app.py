import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and features
model = joblib.load("model/xgb_plant_generation_model.pkl")
features = joblib.load("model/feature_list.pkl")

st.set_page_config(page_title="Plant Generation Predictor", page_icon="⚡")
st.title("⚡ Power Plant Generation Predictor")
st.markdown("Predict annual electricity generation (GWh) based on plant attributes.")

# Sidebar
st.sidebar.header("🧾 Input Plant Details")

# Example UI logic: Customize based on your actual features
input_data = {}

for feat in features:
    if "fuel" in feat.lower():
        input_data[feat] = st.sidebar.selectbox(f"{feat}", options=[
            "Coal", "Oil", "Gas", "Hydro", "Nuclear", "Biomass", "Wind", "Solar", "Geothermal", "Waste", "Other"
        ])
    elif "latitude" in feat.lower():
        input_data[feat] = st.sidebar.slider("Latitude", min_value=-90.0, max_value=90.0, step=0.01)
    elif "longitude" in feat.lower():
        input_data[feat] = st.sidebar.slider("Longitude", min_value=-180.0, max_value=180.0, step=0.01)
    elif "year" in feat.lower():
        input_data[feat] = st.sidebar.number_input(f"{feat}", min_value=1900, max_value=2025, step=1)
    elif "capacity" in feat.lower():
        input_data[feat] = st.sidebar.number_input("Plant Capacity (MW)", min_value=1.0, max_value=25000.0, step=1.0)
    else:
        input_data[feat] = st.sidebar.number_input(f"{feat}", step=0.01)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict button
if st.sidebar.button("🔮 Predict Generation"):
    pred = model.predict(input_df)[0]
    st.success(f"📈 Predicted Annual Generation: **{pred:.2f} GWh**")

    st.markdown("#### 🔍 Input Summary")
    st.dataframe(input_df.T, use_container_width=True)
else:
    st.info("💡 Fill in all fields and click **Predict Generation**.")

