import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD ARTIFACTS (CACHED)
# ==========================================
# Caching prevents the model and encoders from reloading on every button click
@st.cache_resource
def load_ml_artifacts():
    try:
        # Load the Random Forest model
        with open('optimized_rf_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            
        # Load the dictionary of label encoders
        with open('label_encoder (1).pkl', 'rb') as le_file:
            label_encoders = pickle.load(le_file)
            
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

model, label_encoders = load_ml_artifacts()

# ==========================================
# 3. APP HEADER & STYLING
# ==========================================
st.title("🍔 Food Delivery ETA Predictor")
st.markdown("""
Welcome to the Delivery Time Prediction app! This tool utilizes an **Optimized Random Forest Regressor** to estimate how many minutes a food delivery will take based on real-time logistical and environmental factors.
""")
st.divider()

# ==========================================
# 4. USER INPUT SECTION (UI LAYOUT)
# ==========================================
st.subheader("📝 Enter Delivery Details")

# Using columns to make the UI look clean and organized
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Logistics Info**")
    distance_km = st.number_input("Distance (in km)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    prep_time = st.number_input("Food Preparation Time (mins)", min_value=0, max_value=120, value=15, step=1)
    courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)

with col2:
    st.markdown("**Environmental Factors**")
    # Extracting classes dynamically from your label encoder dictionary
    weather_options = label_encoders['Weather'].classes_
    traffic_options = label_encoders['Traffic_Level'].classes_
    
    weather = st.selectbox("Weather Condition", options=weather_options)
    traffic_level = st.selectbox("Traffic Level", options=traffic_options)

with col3:
    st.markdown("**Timing & Vehicle**")
    time_options = label_encoders['Time_of_Day'].classes_
    vehicle_options = label_encoders['Vehicle_Type'].classes_
    
    time_of_day = st.selectbox("Time of Day", options=time_options)
    vehicle_type = st.selectbox("Vehicle Type", options=vehicle_options)

st.divider()

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
# Centered predict button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_button = st.button("🚀 Predict Delivery Time", use_container_width=True)

if predict_button:
    # Adding a visual spinner while "calculating"
    with st.spinner('Running data through the Random Forest model...'):
        time.sleep(0.8) # Tiny artificial delay for UI feel
        
        # 1. Encode the categorical inputs using the loaded encoders
        try:
            encoded_weather = label_encoders['Weather'].transform([weather])[0]
            encoded_traffic = label_encoders['Traffic_Level'].transform([traffic_level])[0]
            encoded_time = label_encoders['Time_of_Day'].transform([time_of_day])[0]
            encoded_vehicle = label_encoders['Vehicle_Type'].transform([vehicle_type])[0]
            
            # 2. Create the feature array in the exact order the model expects
            # Order based on your dataset: Distance_km, Weather, Traffic_Level, Time_of_Day, Vehicle_Type, Preparation_Time_min, Courier_Experience_yrs
            input_features = np.array([[
                distance_km,
                encoded_weather,
                encoded_traffic,
                encoded_time,
                encoded_vehicle,
                prep_time,
                courier_exp
            ]])
            
            # 3. Make prediction
            prediction = model.predict(input_features)[0]
            
            # 4. Display Results beautifully
            st.success("Prediction generated successfully!")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric(label="Estimated Delivery Time", value=f"{int(round(prediction))} Minutes")
            
            with res_col2:
                st.info(f"""
                **Analysis Breakdown:**
                * Driving **{distance_km}km** in **{traffic_level.lower()}** traffic.
                * The food will take **{prep_time} mins** to prepare.
                * Your courier has **{courier_exp} years** of experience riding a **{vehicle_type.lower()}**.
                """)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")