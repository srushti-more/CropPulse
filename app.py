import streamlit as st
try:
    from transformers import pipeline
except ImportError:
    # This forces the app to show a helpful message if the install fails again
    st.error("AI modules failed to load. Please check requirements.txt")
from PIL import Image
import json
import pandas as pd
import plotly.express as px
import AutoImageProcessor
from datetime import datetime

# --- MULTILINGUAL DICTIONARY ---
LANG_DATA = {
    "English": {
        "title": "CropPulse: Precision AI",
        "subtitle": "AI-Powered Diagnosis & Resource Optimizer",
        "scan_header": "📸 Scan Crop",
        "results_header": "🔍 AI Diagnosis",
        "treatment_tab": "💊 Treatment Plan",
        "opti_tab": "📉 Smart Savings",
        "expert_header": "👨‍🔬 Consult Expert",
        "history_header": "📈 Farm Health Trend",
        "metrics": ["Water Saved", "Pesticide Saved", "Yield Protection"],
        "lang_label": "Language",
        "field_label": "Field Size (Acres)",
        "expert_btn": "Message Agricultural Expert",
        "expert_success": "Request sent! An expert will contact you at ",
        "cam_toggle": "Power on Camera",
        "input_label": "Select Input Method",
        "history_insight": "Insight: The most frequent condition is",
    },
    "Hindi": {
        "title": "क्रॉपपल्स (CropPulse): सटीक एआई",
        "subtitle": "एआई-संचालित निदान और संसाधन अनुकूलक",
        "scan_header": "📸 फसल स्कैन करें",
        "results_header": "🔍 एआई निदान",
        "treatment_tab": "💊 उपचार योजना",
        "opti_tab": "📉 स्मार्ट बचत",
        "expert_header": "👨‍🔬 विशेषज्ञ से सलाह लें",
        "history_header": "📈 फार्म स्वास्थ्य रुझान",
        "metrics": ["बचाया गया पानी", "बचाया गया कीटनाशक", "पैदावार सुरक्षा"],
        "lang_label": "भाषा",
        "field_label": "खेत का आकार (एकड़)",
        "expert_btn": "कृषि विशेषज्ञ को संदेश भेजें",
        "expert_success": "अनुरोध भेज दिया गया! एक विशेषज्ञ आपसे संपर्क करेगा: ",
        "cam_toggle": "कैमरा चालू करें",
        "input_label": "इनपुट विधि चुनें",
        "history_insight": "सुझाव: सबसे आम स्थिति है",
    }
}

# --- PAGE CONFIG ---
st.set_page_config(page_title="CropPulse AI", page_icon="🌱", layout="wide")

if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# UI Styling
st.markdown("""
    <style>
    .main { background-color: #f0f4f0; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #2e7d32; color: white; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

from transformers import pipeline, AutoImageProcessor

@st.cache_resource
def get_models():
    model_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    
    # 1. Explicitly load the processor
    processor = AutoImageProcessor.from_pretrained(model_id)
    
    # 2. Pass the processor directly to the pipeline
    return pipeline(
        "image-classification", 
        model=model_id,
        image_processor=processor,
        framework="pt"
    )

classifier = get_models()

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2760/2760144.png", width=80)
selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
T = LANG_DATA[selected_lang]
field_size = st.sidebar.number_input(T["field_label"], min_value=0.1, value=1.0, step=0.5)

# --- HEADER ---
st.title(T["title"])
st.markdown(f"*{T['subtitle']}*")

# --- MAIN APP ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(T["scan_header"])
    input_mode = st.radio(T["input_label"], ["File Upload", "Live Camera"])
    
    final_img = None
    if input_mode == "File Upload":
        img_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        if img_file:
            final_img = Image.open(img_file).convert("RGB")
    else:
        use_cam = st.toggle(T["cam_toggle"])
        if use_cam:
            cam_file = st.camera_input("")
            if cam_file:
                final_img = Image.open(cam_file).convert("RGB")

if final_img:
    with col_right:
        st.subheader(T["results_header"])
        with st.spinner("AI analyzing..."):
            predictions = classifier(final_img)
            res = predictions[0][0]
            
            raw_label = res['label']
            clean_label = raw_label.replace("___", " - ").replace("_", " ")
            confidence = res['score']

            st.metric("Condition", clean_label, f"{confidence*100:.1f}% Match")

            # Update History
            now = datetime.now().strftime("%H:%M:%S")
            new_entry = {"Time": now, "Condition": clean_label, "Confidence": round(confidence*100, 1), "Score": round(confidence*10, 1)}
            if not st.session_state.scan_history or st.session_state.scan_history[-1]["Time"] != now:
                st.session_state.scan_history.append(new_entry)

            # --- KNOWLEDGE BASE LOOKUP ---
            try:
                with open('knowledge.json') as f:
                    kb = json.load(f)
                data = kb.get(raw_label) or kb.get(clean_label) or {
                    "symptoms": "Detailed symptoms not found for this Indian crop variant.", 
                    "pesticide": "Consult a local KVK (Krishi Vigyan Kendra) expert.", 
                    "hindi_pest": "स्थानीय कृषि विज्ञान केंद्र विशेषज्ञ से सलाह लें।"
                }
            except:
                data = {"symptoms": "Knowledge base not loaded.", "pesticide": "N/A", "hindi_pest": "त्रुटि"}

            t1, t2, t3 = st.tabs([T["treatment_tab"], T["opti_tab"], T["expert_header"]])
            
            with t1:
                st.info(f"**Symptoms:** {data.get('symptoms')}")
                st.success(f"**Treatment:** {data['pesticide'] if selected_lang == 'English' else data.get('hindi_pest')}")
            
            with t2:
                water_baseline = field_size * 200
                water_saved = water_baseline * (1 - confidence)
                st.write(f"💧 {T['metrics'][0]}: **{water_saved:.1f} Liters**")
                
                chart_df = pd.DataFrame({
                    "Method": ["Traditional", "CropPulse AI"],
                    "Resources (L)": [water_baseline, water_baseline - water_saved]
                })
                fig_bar = px.bar(chart_df, x="Method", y="Resources (L)", color="Method", 
                                 color_discrete_map={"Traditional": "#bdbdbd", "CropPulse AI": "#2e7d32"})
                st.plotly_chart(fig_bar, use_container_width=True)

            with t3:
                with st.form("expert_form"):
                    u_phone = st.text_input("Mobile / फ़ोन")
                    u_msg = st.text_area("Describe issue / समस्या बताएं")
                    if st.form_submit_button(T["expert_btn"]):
                        st.success(f"{T['expert_success']} {u_phone}")

# --- HISTORY SECTION ---
st.divider()
st.subheader(T["history_header"])

if st.session_state.scan_history:
    df_hist = pd.DataFrame(st.session_state.scan_history)
    h_col1, h_col2 = st.columns([2, 1])
    
    with h_col1:
        # INDENTED CORRECTLY: This block is inside 'with h_col1'
        fig_line = px.line(
            df_hist, 
            x="Time", 
            y="Score", 
            markers=True, 
            title="Farm Health Progression Index",
            color_discrete_sequence=['#2e7d32']
        )
        fig_line.update_layout(yaxis_range=[0, 10]) 
        st.plotly_chart(fig_line, use_container_width=True)
    
    with h_col2:
        # INDENTED CORRECTLY: This block is inside 'with h_col2'
        st.write(f"**{T['history_insight']}:**")
        st.warning(df_hist['Condition'].mode()[0])
        st.dataframe(df_hist.tail(5), hide_index=True)
else:
    st.info("Perform a scan to see health history.")

# import streamlit as st
# from transformers import pipeline
# from PIL import Image
# import json
# import pandas as pd
# import plotly.express as px

# # --- MULTILINGUAL DICTIONARY ---
# LANG_DATA = {
#     "English": {
#         "title": "CropPulse: Precision AI",
#         "subtitle": "AI-Powered Diagnosis & Resource Optimizer",
#         "scan_header": "📸 Scan Crop",
#         "results_header": "🔍 AI Diagnosis",
#         "treatment_tab": "💊 Treatment Plan",
#         "opti_tab": "📉 Smart Savings",
#         "expert_header": "👨‍🔬 Consult Expert",
#         "metrics": ["Water Saved", "Pesticide Saved", "Yield Protection"],
#         "lang_label": "Language",
#         "field_label": "Field Size (Acres)",
#         "expert_btn": "Message Agricultural Expert",
#         "expert_success": "Request sent! An expert will contact you at ",
#     },
#     "Hindi": {
#         "title": "क्रॉपपल्स (CropPulse): सटीक एआई",
#         "subtitle": "एआई-संचालित निदान और संसाधन अनुकूलक",
#         "scan_header": "📸 फसल स्कैन करें",
#         "results_header": "🔍 एआई निदान",
#         "treatment_tab": "💊 उपचार योजना",
#         "opti_tab": "📉 स्मार्ट बचत",
#         "expert_header": "👨‍🔬 विशेषज्ञ से सलाह लें",
#         "metrics": ["बचाया गया पानी", "बचाया गया कीटनाशक", "पैदावार सुरक्षा"],
#         "lang_label": "भाषा",
#         "field_label": "खेत का आकार (एकड़)",
#         "expert_btn": "कृषि विशेषज्ञ को संदेश भेजें",
#         "expert_success": "अनुरोध भेज दिया गया! एक विशेषज्ञ आपसे संपर्क करेगा: ",
#     }
# }

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="CropPulse AI", page_icon="🌱", layout="wide")

# # UI Styling
# st.markdown("""
#     <style>
#     .main { background-color: #f0f4f0; }
#     .stButton>button { width: 100%; border-radius: 20px; background-color: #2e7d32; color: white; }
#     .metric-card { background: white; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
#     </style>
#     """, unsafe_allow_html=True)

# @st.cache_resource
# def get_models():
#     return pipeline("image-classification", model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")

# classifier = get_models()

# # --- SIDEBAR & GLOBAL SETTINGS ---
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2760/2760144.png", width=80)
# selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
# T = LANG_DATA[selected_lang]

# field_size = st.sidebar.number_input(T["field_label"], min_value=0.1, value=1.0)

# # --- HEADER ---
# st.title(T["title"])
# st.markdown(f"*{T['subtitle']}*")

# # --- MAIN APP ---
# col_left, col_right = st.columns([1, 1])

# with col_left:
#     st.subheader(T["scan_header"])
#     img_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
#     cam_file = st.camera_input("")
#     final_img = cam_file if cam_file else img_file

# if final_img:
#     img = Image.open(final_img).convert("RGB")
    
#     with col_right:
#         st.subheader(T["results_header"])
#         with st.spinner("AI analyzing..."):
#             predictions = classifier([img])
#             res = predictions[0][0]
#             label = res['label']
#             confidence = res['score']

#             # Display Diagnosis Card
#             st.metric("Condition", label, f"{confidence*100:.1f}%")

#             # Load Knowledge
#             try:
#                 with open('knowledge.json') as f: kb = json.load(f)
#                 data = kb.get(label, {"symptoms": "N/A", "pesticide": "Check manual", "hindi_pest": "विशेषज्ञ से पूछें"})
#             except:
#                 data = {"symptoms": "Error loading DB", "pesticide": "N/A", "hindi_pest": "त्रुटि"}

#             # Recommendations Tabs
#             t1, t2, t3 = st.tabs([T["treatment_tab"], T["opti_tab"], T["expert_header"]])
            
#             with t1:
#                 st.write(f"**Symptoms:** {data.get('symptoms', '...')}")
#                 st.success(f"**Recommended:** {data['pesticide'] if selected_lang == 'English' else data.get('hindi_pest', '...')}")
            
#             with t2:
#                 # DYNAMIC MATH BASED ON FIELD SIZE & CONFIDENCE
#                 # Formula: Savings = Field_Size * Confidence_Factor
#                 water_saved = field_size * 120 * confidence 
#                 pest_saved = field_size * 2.5 * confidence
                
#                 st.info(f"By targeting only the infected **{label}** clusters:")
#                 st.write(f"💧 {T['metrics'][0]}: **{water_saved:.1f} Liters**")
#                 st.write(f"🧪 {T['metrics'][1]}: **{pest_saved:.2f} kg**")

#             with t3:
#                 with st.form("expert_form"):
#                     u_phone = st.text_input("Mobile / फ़ोन")
#                     u_msg = st.text_area("Describe issue / समस्या बताएं")
#                     if st.form_submit_button(T["expert_btn"]):
#                         st.balloons()
#                         st.success(f"{T['expert_success']} {u_phone}")

# # --- ANALYTICS SECTION ---
# st.divider()
# st.subheader("📊 " + ("Real-time Impact Map" if selected_lang == "English" else "वास्तविक समय प्रभाव मानचित्र"))

# # Dynamic Graph based on field size
# m1, m2, m3 = st.columns(3)
# # Logic: If field is larger, yield impact looks bigger
# potential_yield = 15 + (field_size * 0.5)

# m1.metric(T["metrics"][0], f"{field_size * 450:.0f} L")
# m2.metric(T["metrics"][1], f"{field_size * 1.2:.1f} kg")
# m3.metric(T["metrics"][2], f"↑ {potential_yield:.1f}%")

# # Sample chart
# chart_data = pd.DataFrame({
#     'Metric': ['Chemicals', 'Water', 'Labor'],
#     'Traditional': [100, 100, 100],
#     'CropPulse': [100 - (confidence*40), 100 - (confidence*30), 80]
# })
# fig = px.bar(chart_data, x='Metric', y=['Traditional', 'CropPulse'], barmode='group', color_discrete_sequence=['#bdbdbd', '#2e7d32'])
# st.plotly_chart(fig, width='stretch')
