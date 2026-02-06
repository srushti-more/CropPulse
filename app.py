import streamlit as st
try:
    from transformers import pipeline, AutoImageProcessor
except ImportError:
    st.error("AI modules failed to load. Please check requirements.txt")
from PIL import Image
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- MULTILINGUAL DICTIONARY ---
LANG_DATA = {
    "English": {
        "title": "CropPulse: Precision AI",
        "subtitle": "Indian Smart Farming & Resource Optimizer",
        "scan_header": "📸 Scan Crop",
        "results_header": "🔍 AI Diagnosis",
        "treatment_tab": "💊 Treatment Plan",
        "impact_tab": "📊 Impact Analysis",
        "expert_header": "👨‍🔬 Consult Expert",
        "history_header": "📈 Farm Health Trend",
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
        "subtitle": "भारतीय स्मार्ट खेती और संसाधन अनुकूलक",
        "scan_header": "📸 फसल स्कैन करें",
        "results_header": "🔍 एआई निदान",
        "treatment_tab": "💊 उपचार योजना",
        "impact_tab": "📊 प्रभाव विश्लेषण",
        "expert_header": "👨‍🔬 विशेषज्ञ से सलाह लें",
        "history_header": "📈 फार्म स्वास्थ्य रुझान",
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

# --- CUSTOM CSS (Glassmorphism & Farm Background) ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                    url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        margin-bottom: 20px;
    }
    h1, h2, h3, p, label, .stMarkdown { color: white !important; }
    .stMetric { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px; color: white !important; }
    div[data-testid="stMetricValue"] > div { color: #90EE90 !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_models():
    model_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    # Explicitly load processor to fix ValueError and ensure proper scaling
    processor = AutoImageProcessor.from_pretrained(model_id)
    return pipeline(
        "image-classification", 
        model=model_id,
        image_processor=processor,
        framework="pt"
    )

classifier = get_models()

# --- SIDEBAR ---
st.sidebar.markdown("### 🚜 Farm Dashboard")
selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
T = LANG_DATA[selected_lang]
field_size = st.sidebar.slider(T["field_label"], 0.5, 50.0, 5.0)
crop_type = st.sidebar.selectbox("Crop Type", ["Rice", "Wheat", "Sugarcane", "Potato", "Tomato"])

# --- HEADER ---
st.title(T["title"])
st.markdown(f"### *{T['subtitle']}*")

# --- MAIN APP ---
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(T["scan_header"])
    input_mode = st.radio(T["input_label"], ["File Upload", "Live Camera"], horizontal=True)
    
    final_img = None
    if input_mode == "File Upload":
        img_file = st.file_uploader(T["scan_header"], type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if img_file:
            final_img = Image.open(img_file).convert("RGB")
            st.image(final_img, caption="Uploaded Image", use_container_width=True)
    else:
        use_cam = st.toggle(T["cam_toggle"])
        if use_cam:
            cam_file = st.camera_input(T["scan_header"], label_visibility="collapsed")
            if cam_file:
                final_img = Image.open(cam_file).convert("RGB")
    st.markdown('</div>', unsafe_allow_html=True)

if final_img:
    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(T["results_header"])
        with st.spinner("AI analyzing pathogens..."):
            predictions = classifier(final_img)
            # Pick the top prediction dynamically
            res = max(predictions, key=lambda x: x['score'])
            
            raw_label = res['label']
            clean_label = raw_label.replace("___", " - ").replace("_", " ")
            confidence = res['score']

            st.metric("Condition", clean_label, f"{confidence*100:.1f}% Match")

            # Precision Actionable Logic
            # affected_area is a heuristic: lower confidence suggests more uncertainty/larger potential spread
            affected_area_pct = max(5, round((1.1 - confidence) * 100)) 
            chem_saved_pct = 100 - affected_area_pct
            yield_prot_pct = round(confidence * 100, 1)

            m1, m2, m3 = st.columns(3)
            m1.write(f"📉 **Spot Spraying:**\n{affected_area_pct}% of field")
            m2.write(f"🧪 **Chem Savings:**\n{chem_saved_pct}% saved")
            m3.write(f"🌾 **Yield Protected:**\n{yield_prot_pct}%")

            # Update History
            now = datetime.now().strftime("%H:%M:%S")
            new_entry = {"Time": now, "Condition": clean_label, "Score": round(confidence*10, 1)}
            if not st.session_state.scan_history or st.session_state.scan_history[-1]["Time"] != now:
                st.session_state.scan_history.append(new_entry)

            # --- KNOWLEDGE BASE LOOKUP ---
            try:
                with open('knowledge.json') as f:
                    kb = json.load(f)
                data = kb.get(raw_label) or kb.get(clean_label) or {
                    "symptoms": "Symptoms not found.", 
                    "pesticide": "Consult expert.", 
                    "organic": "Use Neem oil spray.",
                    "hindi_pest": "विशेषज्ञ से सलाह लें।"
                }
            except:
                data = {"symptoms": "Error loading data.", "pesticide": "N/A", "organic": "N/A"}

            tab1, tab2, tab3 = st.tabs([T["treatment_tab"], T["impact_tab"], T["expert_header"]])
            
            with tab1:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
                    <p><b>Symptoms:</b> {data.get('symptoms')}</p>
                    <h4 style="color: #FFD700;">🧪 Chemical Dosage</h4>
                    <p>{data['pesticide'] if selected_lang == 'English' else data.get('hindi_pest')}</p>
                    <h4 style="color: #90EE90;">🌿 Desi (Organic) Hack</h4>
                    <p>{data.get('organic', 'Apply 10% Neem Oil solution.')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with tab2:
                impact_df = pd.DataFrame({
                    "Category": ["Pesticide Waste", "Chemical Saved"],
                    "Liters": [affected_area_pct * field_size, chem_saved_pct * field_size]
                })
                fig = px.pie(impact_df, values="Liters", names="Category", hole=0.5,
                             color_discrete_sequence=['#ff4b4b', '#2e7d32'])
                fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Saving **{chem_saved_pct * field_size:.1f}L** of chemicals on your {field_size} acre farm.")

            with tab3:
                with st.form("expert_form"):
                    u_phone = st.text_input("Mobile / फ़ोन")
                    u_msg = st.text_area("Describe issue / समस्या बताएं")
                    if st.form_submit_button(T["expert_btn"]):
                        st.success(f"{T['expert_success']} {u_phone}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- HISTORY SECTION ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader(T["history_header"])

if st.session_state.scan_history:
    df_hist = pd.DataFrame(st.session_state.scan_history)
    h_col1, h_col2 = st.columns([2, 1])
    
    with h_col1:
        fig_line = px.line(df_hist, x="Time", y="Score", markers=True, color_discrete_sequence=['#90EE90'])
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", yaxis_range=[0, 10])
        st.plotly_chart(fig_line, use_container_width=True)
    
    with h_col2:
        st.write(f"**{T['history_insight']}:**")
        st.warning(df_hist['Condition'].mode()[0])
        st.dataframe(df_hist.tail(3), hide_index=True)
else:
    st.info("Perform a scan to see health history.")
st.markdown('</div>', unsafe_allow_html=True)

# import streamlit as st
# try:
#     from transformers import pipeline, AutoImageProcessor
# except ImportError:
#     st.error("AI modules failed to load. Please check requirements.txt")
# from PIL import Image
# import json
# import pandas as pd
# import plotly.express as px
# from datetime import datetime

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
#         "history_header": "📈 Farm Health Trend",
#         "metrics": ["Water Saved", "Pesticide Saved", "Yield Protection"],
#         "lang_label": "Language",
#         "field_label": "Field Size (Acres)",
#         "expert_btn": "Message Agricultural Expert",
#         "expert_success": "Request sent! An expert will contact you at ",
#         "cam_toggle": "Power on Camera",
#         "input_label": "Select Input Method",
#         "history_insight": "Insight: The most frequent condition is",
#     },
#     "Hindi": {
#         "title": "क्रॉपपल्स (CropPulse): सटीक एआई",
#         "subtitle": "एआई-संचालित निदान और संसाधन अनुकूलक",
#         "scan_header": "📸 फसल स्कैन करें",
#         "results_header": "🔍 एआई निदान",
#         "treatment_tab": "💊 उपचार योजना",
#         "opti_tab": "📉 स्मार्ट बचत",
#         "expert_header": "👨‍🔬 विशेषज्ञ से सलाह लें",
#         "history_header": "📈 फार्म स्वास्थ्य रुझान",
#         "metrics": ["बचाया गया पानी", "बचाया गया कीटनाशक", "पैदावार सुरक्षा"],
#         "lang_label": "भाषा",
#         "field_label": "खेत का आकार (एकड़)",
#         "expert_btn": "कृषि विशेषज्ञ को संदेश भेजें",
#         "expert_success": "अनुरोध भेज दिया गया! एक विशेषज्ञ आपसे संपर्क करेगा: ",
#         "cam_toggle": "कैमरा चालू करें",
#         "input_label": "इनपुट विधि चुनें",
#         "history_insight": "सुझाव: सबसे आम स्थिति है",
#     }
# }

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="CropPulse", page_icon="🌱", layout="wide")

# if 'scan_history' not in st.session_state:
#     st.session_state.scan_history = []

# # UI Styling
# st.markdown("""
#     <style>
#     .main { background-color: #f0f4f0; }
#     .stButton>button { width: 100%; border-radius: 20px; background-color: #2e7d32; color: white; }
#     .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); }
#     </style>
#     """, unsafe_allow_html=True)

# @st.cache_resource
# def get_models():
#     model_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
#     # This processor ensures the image is resized to 224x224 and colors are normalized
#     processor = AutoImageProcessor.from_pretrained(model_id)
#     return pipeline(
#         "image-classification", 
#         model=model_id,
#         image_processor=processor,
#         framework="pt"
#     )

# classifier = get_models()

# # --- SIDEBAR ---
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2760/2760144.png", width=80)
# selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
# T = LANG_DATA[selected_lang]
# field_size = st.sidebar.number_input(T["field_label"], min_value=0.1, value=1.0, step=0.5)

# # --- HEADER ---
# st.title(T["title"])
# st.markdown(f"*{T['subtitle']}*")

# # --- MAIN APP ---
# col_left, col_right = st.columns([1, 1])

# with col_left:
#     st.subheader(T["scan_header"])
#     input_mode = st.radio(T["input_label"], ["File Upload", "Live Camera"])
    
#     final_img = None
#     if input_mode == "File Upload":
#         img_file = st.file_uploader(
#             T["scan_header"], 
#             type=["jpg", "png", "jpeg"], 
#             label_visibility="collapsed"
#         )
#         if img_file:
#             final_img = Image.open(img_file).convert("RGB")
#     else:
#         use_cam = st.toggle(T["cam_toggle"])
#         if use_cam:
#             cam_file = st.camera_input(T["scan_header"], label_visibility="collapsed")
#             if cam_file:
#                 final_img = Image.open(cam_file).convert("RGB")

# if final_img:
#     with col_right:
#         st.subheader(T["results_header"])
#         with st.spinner("AI analyzing..."):
#             predictions = classifier(final_img)

#             top_prediction = max(predictions, key=lambda x: x['score'])
            
#             raw_label = top_prediction['label']
#             confidence = top_prediction['score']
#             # Standardizing result extraction
#             res = predictions[0] if isinstance(predictions[0], dict) else predictions[0][0]
            
#             raw_label = res['label']
#             clean_label = raw_label.replace("___", " - ").replace("_", " ")
#             confidence = res['score']

#             st.metric("Condition", clean_label, f"{confidence*100:.1f}% Match")

#             # Update History
#             now = datetime.now().strftime("%H:%M:%S")
#             new_entry = {"Time": now, "Condition": clean_label, "Confidence": round(confidence*100, 1), "Score": round(confidence*10, 1)}
#             if not st.session_state.scan_history or st.session_state.scan_history[-1]["Time"] != now:
#                 st.session_state.scan_history.append(new_entry)

#             # --- KNOWLEDGE BASE LOOKUP ---
#             try:
#                 with open('knowledge.json') as f:
#                     kb = json.load(f)
#                 data = kb.get(raw_label) or kb.get(clean_label) or {
#                     "symptoms": "Detailed symptoms not found for this Indian crop variant.", 
#                     "pesticide": "Consult a local KVK (Krishi Vigyan Kendra) expert.", 
#                     "hindi_pest": "स्थानीय कृषि विज्ञान केंद्र विशेषज्ञ से सलाह लें।"
#                 }
#             except:
#                 data = {"symptoms": "Knowledge base not loaded.", "pesticide": "N/A", "hindi_pest": "त्रुटि"}

#             t1, t2, t3 = st.tabs([T["treatment_tab"], T["opti_tab"], T["expert_header"]])
            
#             with t1:
#                 st.info(f"**Symptoms:** {data.get('symptoms')}")
#                 st.success(f"**Treatment:** {data['pesticide'] if selected_lang == 'English' else data.get('hindi_pest')}")
            
#             with t2:
#                 water_baseline = field_size * 200
#                 water_saved = water_baseline * (1 - confidence)
#                 st.write(f"💧 {T['metrics'][0]}: **{water_saved:.1f} Liters**")
                
#                 chart_df = pd.DataFrame({
#                     "Method": ["Traditional", "CropPulse AI"],
#                     "Resources (L)": [water_baseline, water_baseline - water_saved]
#                 })
#                 fig_bar = px.bar(chart_df, x="Method", y="Resources (L)", color="Method", 
#                                  color_discrete_map={"Traditional": "#bdbdbd", "CropPulse AI": "#2e7d32"})
#                 st.plotly_chart(fig_bar, use_container_width=True)

#             with t3:
#                 with st.form("expert_form"):
#                     u_phone = st.text_input("Mobile / फ़ोन")
#                     u_msg = st.text_area("Describe issue / समस्या बताएं")
#                     if st.form_submit_button(T["expert_btn"]):
#                         st.success(f"{T['expert_success']} {u_phone}")

# # --- HISTORY SECTION ---
# st.divider()
# st.subheader(T["history_header"])

# if st.session_state.scan_history:
#     df_hist = pd.DataFrame(st.session_state.scan_history)
#     h_col1, h_col2 = st.columns([2, 1])
    
#     with h_col1:
#         fig_line = px.line(
#             df_hist, 
#             x="Time", 
#             y="Score", 
#             markers=True, 
#             title="Farm Health Progression Index",
#             color_discrete_sequence=['#2e7d32']
#         )
#         fig_line.update_layout(yaxis_range=[0, 10]) 
#         st.plotly_chart(fig_line, use_container_width=True)
    
#     with h_col2:
#         st.write(f"**{T['history_insight']}:**")
#         st.warning(df_hist['Condition'].mode()[0])
#         st.dataframe(df_hist.tail(5), hide_index=True)
# else:
#     st.info("Perform a scan to see health history.")

