import streamlit as st
from transformers import pipeline
from PIL import Image
import json
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CropPulse", page_icon="🌱", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def get_models():
    # Using the specific MobileNetV2 model
    model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    return pipeline("image-classification", model=model_name)

classifier = get_models()

# --- HEADER ---
st.title("🌿 CropPulse: Precision Agriculture")
st.markdown("### *AI-Powered Pest Detection & Pesticide Optimizer*")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    lang = st.selectbox("Language / भाषा", ["English", "Hindi"])
    field_size = st.number_input("Field Size (Acres)", min_value=0.1, value=1.0)

# --- MAIN INTERFACE ---
col_input, col_results = st.columns([1, 1])

with col_input:
    st.subheader("📸 Scan Crop")
    img_file = st.file_uploader("Upload leaf image...", type=["jpg", "png", "jpeg"])
    # If camera is used, it replaces the uploader
    cam_file = st.camera_input("Or take a photo")
    
    final_img = cam_file if cam_file else img_file

if final_img:
    # IMPORTANT: Force RGB to avoid transparency/alpha channel issues
    img = Image.open(final_img).convert("RGB")
    
    with col_results:
        st.subheader("🔍 AI Diagnosis")
        with st.spinner("Analyzing tissue health..."):
            # FIX: Passing as a list [img] prevents the 'Unable to create tensor' error
            predictions = classifier([img])
            res = predictions[0][0] # Get top result from the batch
            label = res['label']
            score = res['score']

            st.metric("Detected Condition", label, f"{score*100:.1f}% Match")

            # Load Knowledge Base
            try:
                with open('knowledge.json') as f:
                    kb = json.load(f)
                data = kb.get(label, {"symptoms": "Check moisture levels.", "pesticide": "Consult expert", "dosage": "N/A", "optimization": "N/A", "hindi": "अज्ञात"})
            except:
                data = {"symptoms": "Data missing.", "pesticide": "N/A", "dosage": "N/A", "optimization": "N/A", "hindi": "त्रुटि"}

            st.warning(f"**Symptoms:** {data['symptoms']}")
            
            t1, t2 = st.tabs(["💊 Treatment", "📉 Optimization"])
            with t1:
                st.success(f"**Recommended:** {data['pesticide'] if lang == 'English' else data['hindi']}")
            with t2:
                st.info(f"**Spray Efficiency:** {field_size * 0.25:.2f} Acres Targeted")

# --- ANALYTICS ---
st.divider()
st.subheader("📊 Farm Health Analytics")
c1, c2 = st.columns(2)
with c1:
    df = pd.DataFrame({'Day': ['M', 'T', 'W', 'T', 'F'], 'Pests': [0.8, 0.6, 0.4, 0.2, 0.1]})
    fig = px.area(df, x='Day', y='Pests')
    # Use width='stretch' to satisfy the 2026 Streamlit requirement
    st.plotly_chart(fig, width='stretch') 

with c2:
    st.table(pd.DataFrame({"Metric": ["Water Saved", "Yield ↑"], "Value": ["1.2k L", "18%"]}))