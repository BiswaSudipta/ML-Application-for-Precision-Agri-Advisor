import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. App Configuration & Branding
# ==========================================
st.set_page_config(
    page_title="UPL NextGen Agri-Advisor",
    page_icon="🌱",
    layout="centered"
)

# Custom CSS to make it look professional (UPL Colors)
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #ff914d;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #e67e3e;
    }
    h1 {
        color: #2E8B57;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Load Model
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        artifacts = joblib.load('crop_recommendation_model.pkl')
        return artifacts
    except FileNotFoundError:
        return None

artifacts = load_artifacts()

# ==========================================
# 3. Sidebar - Input Parameters
# ==========================================
st.sidebar.header("📝 Soil & Climate Input")
st.sidebar.markdown("Adjust the values based on soil test report.")

def user_input_features():
    # Sliders and Inputs
    N = st.sidebar.slider('Nitrogen (N) [kg/ha]', 0, 140, 40)
    P = st.sidebar.slider('Phosphorus (P) [kg/ha]', 0, 145, 50)
    K = st.sidebar.slider('Potassium (K) [kg/ha]', 0, 205, 50)
    
    st.sidebar.markdown("---")
    
    temp = st.sidebar.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.sidebar.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=71.0)
    ph = st.sidebar.number_input('Soil pH', min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.sidebar.number_input('Rainfall (mm)', min_value=0.0, max_value=300.0, value=100.0)
    
    return N, P, K, temp, humidity, ph, rainfall

N, P, K, temp, humidity, ph, rainfall = user_input_features()

# ==========================================
# 4. Main Page Content
# ==========================================
st.title("🌱 Precision Agri-Input Advisor")
st.markdown("### AI-Driven Crop Recommendation System")
st.markdown("""
This tool helps agronomists and farmers identify the **optimal crop** for specific soil conditions, 
ensuring sustainable resource usage and maximum yield efficiency.
""")

# Display Inputs inputs in a nice grid
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nitrogen", f"{N}", "kg/ha")
col2.metric("Phosphorus", f"{P}", "kg/ha")
col3.metric("Potassium", f"{K}", "kg/ha")
col4.metric("pH Level", f"{ph}")

# ==========================================
# 5. Prediction Logic
# ==========================================
if artifacts:
    model = artifacts['model']
    le = artifacts['label_encoder']
    feature_names = artifacts['features'] # List of feature names the model expects

    if st.button('🔍 Recommend Best Crop'):
        with st.spinner('Analyzing soil parameters...'):
            # Feature Engineering (Must match training script)
            total_nutrients = N + P + K
            # Avoid division by zero
            safe_total = total_nutrients if total_nutrients > 0 else 1
            n_ratio = N / safe_total
            p_ratio = P / safe_total
            k_ratio = K / safe_total
            
            # Prepare Dataframe with columns in exact order
            input_data = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall, 
                                      total_nutrients, n_ratio, p_ratio, k_ratio]], 
                                      columns=feature_names)
            
            # Predict
            prediction_idx = model.predict(input_data)[0]
            prediction_name = le.inverse_transform([prediction_idx])[0]
            
            # Probability/Confidence
            proba = model.predict_proba(input_data)
            confidence = np.max(proba) * 100

        # Result Display
        st.success(f"✅ Recommended Crop: **{prediction_name.upper()}**")
        
        # Additional Insights (Business Logic)
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.info(f"Confidence Score: **{confidence:.2f}%**")
        with col_res2:
            st.warning("Action: Consult local agricultural experts for specific seed varieties.")
            
        # Contextual advice based on crop
        st.markdown("#### 🚜 Agronomist Notes:")
        if prediction_name in ['rice', 'jute', 'coconut']:
            st.write(f"This crop requires **High Moisture**. Ensure irrigation channels are prepared.")
        elif prediction_name in ['chickpea', 'lentil', 'kidneybeans']:
            st.write(f"This is a **Nitrogen-fixing crop**. It will naturally replenish soil N levels.")
        else:
            st.write("Ensure standard pest management protocols are followed.")

else:
    st.error("Error: Model file not found. Please run 'train_model.py' first.")

