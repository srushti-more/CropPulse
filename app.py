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
        "subtitle": "Advanced Disease Detection & Resource Management",
        "scan_header": "📸 Scan Crop",
        "results_header": "🔍 AI Analysis Results",
        "treatment_tab": "💊 Treatment Plan",
        "impact_tab": "📉 Smart Savings",
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
        "subtitle": "उन्नत रोग पहचान और संसाधन प्रबंधन",
        "scan_header": "📸 फसल स्कैन करें",
        "results_header": "🔍 एआई विश्लेषण परिणाम",
        "treatment_tab": "💊 उपचार योजना",
        "impact_tab": "📉 स्मार्ट बचत",
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

# --- CUSTOM CSS (Enhanced Blur, Darker Background & Visibility) ---
st.markdown("""
    <style>
    .stApp {
        /* Darker gradient for better text contrast */
        background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)), 
                    url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Solid Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 20, 15, 0.85); 
        backdrop-filter: blur(25px) saturate(160%); 
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 15px 35px rgba(0,0,0,0.6);
        margin-bottom: 25px;
        color: white;
    }
    
    h1, h2, h3 {
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    /* Metric Highlighting */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(144, 238, 144, 0.2) !important;
        border-radius: 15px !important;
    }
    
    div[data-testid="stMetricValue"] > div { color: #90EE90 !important; font-weight: bold; }

    .stRadio label { color: white !important; font-weight: bold; }
    
    /* Improved Uploader Visibility */
    .stFileUploader section {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px dashed rgba(255,255,255,0.3) !important;
        border-radius: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def get_models():
    model_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    processor = AutoImageProcessor.from_pretrained(model_id)
    return pipeline("image-classification", model=model_id, image_processor=processor, framework="pt")

classifier = get_models()

# --- SIDEBAR ---
st.sidebar.markdown("## 🚜 Farm Control Center")
selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
T = LANG_DATA[selected_lang]
field_size = st.sidebar.slider(T["field_label"], 0.5, 50.0, 5.0)
crop_type = st.sidebar.selectbox("Crop Type", ["Rice", "Wheat", "Sugarcane", "Potato", "Tomato"])

# --- HEADER ---
st.title(T["title"])
st.markdown(f"##### *{T['subtitle']}*")

# --- MAIN APP LAYOUT ---
col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(T["scan_header"])
    input_mode = st.radio(T["input_label"], ["File Upload", "Live Camera"], horizontal=True)
    
    final_img = None
    if input_mode == "File Upload":
        img_file = st.file_uploader(T["scan_header"], type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if img_file:
            final_img = Image.open(img_file).convert("RGB")
            st.image(final_img, caption="Analyzed Specimen", use_container_width=True)
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
        with st.spinner("Decoding Plant Pathology..."):
            predictions = classifier(final_img)
            res = max(predictions, key=lambda x: x['score'])
            
            raw_label = res['label']
            clean_label = raw_label.replace("___", " - ").replace("_", " ")
            confidence = res['score']

            # --- CONFIDENCE GUARDRAIL (Prevents Wrong Diagnosis) ---
            if confidence < 0.50:
                st.error("⚠️ **Low Confidence Detection**")
                st.warning(f"AI Match is only {confidence*100:.1f}%. This might be insect damage or a rare variant.")
                st.info("💡 **Observation:** If you see 'skeletonized' leaves (holes between veins), this is likely **Insect Damage**, not a disease. Try applying Neem Oil.")
            else:
                st.metric("Detected Condition", clean_label, f"{confidence*100:.1f}% Match")

            # Precision Actionable Logic
            affected_area = max(5, round((1.1 - confidence) * 100)) 
            chem_saved = 100 - affected_area
            yield_prot = round(confidence * 100, 1)

            # Feature Highlights (Shown only if confidence is usable)
            if confidence > 0.30:
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"🎯 **Target Zone**\n### {affected_area}%")
                    st.caption("Apply treatment here")
                with m2:
                    st.markdown(f"🛡️ **Yield Saved**\n### {yield_prot}%")
                    st.caption("Harvest protected")
                with m3:
                    st.markdown(f"💰 **Cost Saved**\n### {chem_saved}%")
                    st.caption("Lower input cost")

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
                    "symptoms": "Detailed symptoms pending for this variant.", 
                    "pesticide": "Consult your local Krishi Vigyan Kendra.", 
                    "organic": "Apply diluted Neem Oil (5ml/L).",
                    "hindi_pest": "स्थानीय कृषि केंद्र से सलाह लें।"
                }
            except:
                data = {"symptoms": "Database error.", "pesticide": "N/A", "organic": "N/A"}

            tab1, tab2, tab3 = st.tabs([T["treatment_tab"], T["impact_tab"], T["expert_header"]])
            
            with tab1:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                    <p><b>Symptoms:</b> {data.get('symptoms')}</p>
                    <h4 style="color: #FFD700;">🧪 Chemical Dosage</h4>
                    <p>{data['pesticide'] if selected_lang == 'English' else data.get('hindi_pest')}</p>
                    <h4 style="color: #90EE90;">🌿 Desi (Organic) Hack</h4>
                    <p>{data.get('organic')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("#### Precision Resource Allocation")
                impact_df = pd.DataFrame({
                    "Sector": ["Treatment Area", "Protected Area"],
                    "Size": [affected_area, chem_saved]
                })
                fig = px.pie(impact_df, values="Size", names="Sector", hole=0.6,
                             color_discrete_sequence=['#FF4B4B', '#00C851'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False, height=200, margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Estimated **{chem_saved * field_size * 0.1:.1f} kg** chemical reduction for your farm.")

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
#         "subtitle": "Indian Smart Farming & Resource Optimizer",
#         "scan_header": "📸 Scan Crop",
#         "results_header": "🔍 AI Diagnosis",
#         "treatment_tab": "💊 Treatment Plan",
#         "impact_tab": "📊 Impact Analysis",
#         "expert_header": "👨‍🔬 Consult Expert",
#         "history_header": "📈 Farm Health Trend",
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
#         "subtitle": "भारतीय स्मार्ट खेती और संसाधन अनुकूलक",
#         "scan_header": "📸 फसल स्कैन करें",
#         "results_header": "🔍 एआई निदान",
#         "treatment_tab": "💊 उपचार योजना",
#         "impact_tab": "📊 प्रभाव विश्लेषण",
#         "expert_header": "👨‍🔬 विशेषज्ञ से सलाह लें",
#         "history_header": "📈 फार्म स्वास्थ्य रुझान",
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
# st.set_page_config(page_title="CropPulse AI", page_icon="🌱", layout="wide")

# if 'scan_history' not in st.session_state:
#     st.session_state.scan_history = []

# # --- CUSTOM CSS (Glassmorphism & Farm Background) ---
# st.markdown("""
#     <style>
#     .stApp {
#         background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
#                     url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80");
#         background-size: cover;
#         background-attachment: fixed;
#     }
#     .glass-card {
#         background: rgba(255, 255, 255, 0.1);
#         backdrop-filter: blur(10px);
#         border-radius: 15px;
#         padding: 25px;
#         border: 1px solid rgba(255, 255, 255, 0.2);
#         color: white;
#         margin-bottom: 20px;
#     }
#     h1, h2, h3, p, label, .stMarkdown { color: white !important; }
#     .stMetric { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px; color: white !important; }
#     div[data-testid="stMetricValue"] > div { color: #90EE90 !important; }
#     </style>
#     """, unsafe_allow_html=True)

# @st.cache_resource
# def get_models():
#     model_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
#     # Explicitly load processor to fix ValueError and ensure proper scaling
#     processor = AutoImageProcessor.from_pretrained(model_id)
#     return pipeline(
#         "image-classification", 
#         model=model_id,
#         image_processor=processor,
#         framework="pt"
#     )

# classifier = get_models()

# # --- SIDEBAR ---
# st.sidebar.markdown("### 🚜 Farm Dashboard")
# selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
# T = LANG_DATA[selected_lang]
# field_size = st.sidebar.slider(T["field_label"], 0.5, 50.0, 5.0)
# crop_type = st.sidebar.selectbox("Crop Type", ["Rice", "Wheat", "Sugarcane", "Potato", "Tomato"])

# # --- HEADER ---
# st.title(T["title"])
# st.markdown(f"### *{T['subtitle']}*")

# # --- MAIN APP ---
# col_left, col_right = st.columns([1, 1.2])

# with col_left:
#     st.markdown('<div class="glass-card">', unsafe_allow_html=True)
#     st.subheader(T["scan_header"])
#     input_mode = st.radio(T["input_label"], ["File Upload", "Live Camera"], horizontal=True)
    
#     final_img = None
#     if input_mode == "File Upload":
#         img_file = st.file_uploader(T["scan_header"], type=["jpg", "png", "jpeg"], label_visibility="collapsed")
#         if img_file:
#             final_img = Image.open(img_file).convert("RGB")
#             st.image(final_img, caption="Uploaded Image", use_container_width=True)
#     else:
#         use_cam = st.toggle(T["cam_toggle"])
#         if use_cam:
#             cam_file = st.camera_input(T["scan_header"], label_visibility="collapsed")
#             if cam_file:
#                 final_img = Image.open(cam_file).convert("RGB")
#     st.markdown('</div>', unsafe_allow_html=True)

# if final_img:
#     with col_right:
#         st.markdown('<div class="glass-card">', unsafe_allow_html=True)
#         st.subheader(T["results_header"])
#         with st.spinner("AI analyzing pathogens..."):
#             predictions = classifier(final_img)
#             # Pick the top prediction dynamically
#             res = max(predictions, key=lambda x: x['score'])
            
#             raw_label = res['label']
#             clean_label = raw_label.replace("___", " - ").replace("_", " ")
#             confidence = res['score']

#             st.metric("Condition", clean_label, f"{confidence*100:.1f}% Match")

#             # Precision Actionable Logic
#             # affected_area is a heuristic: lower confidence suggests more uncertainty/larger potential spread
#             affected_area_pct = max(5, round((1.1 - confidence) * 100)) 
#             chem_saved_pct = 100 - affected_area_pct
#             yield_prot_pct = round(confidence * 100, 1)

#             m1, m2, m3 = st.columns(3)
#             m1.write(f"📉 **Spot Spraying:**\n{affected_area_pct}% of field")
#             m2.write(f"🧪 **Chem Savings:**\n{chem_saved_pct}% saved")
#             m3.write(f"🌾 **Yield Protected:**\n{yield_prot_pct}%")

#             # Update History
#             now = datetime.now().strftime("%H:%M:%S")
#             new_entry = {"Time": now, "Condition": clean_label, "Score": round(confidence*10, 1)}
#             if not st.session_state.scan_history or st.session_state.scan_history[-1]["Time"] != now:
#                 st.session_state.scan_history.append(new_entry)

#             # --- KNOWLEDGE BASE LOOKUP ---
#             try:
#                 with open('knowledge.json') as f:
#                     kb = json.load(f)
#                 data = kb.get(raw_label) or kb.get(clean_label) or {
#                     "symptoms": "Symptoms not found.", 
#                     "pesticide": "Consult expert.", 
#                     "organic": "Use Neem oil spray.",
#                     "hindi_pest": "विशेषज्ञ से सलाह लें।"
#                 }
#             except:
#                 data = {"symptoms": "Error loading data.", "pesticide": "N/A", "organic": "N/A"}

#             tab1, tab2, tab3 = st.tabs([T["treatment_tab"], T["impact_tab"], T["expert_header"]])
            
#             with tab1:
#                 st.markdown(f"""
#                 <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
#                     <p><b>Symptoms:</b> {data.get('symptoms')}</p>
#                     <h4 style="color: #FFD700;">🧪 Chemical Dosage</h4>
#                     <p>{data['pesticide'] if selected_lang == 'English' else data.get('hindi_pest')}</p>
#                     <h4 style="color: #90EE90;">🌿 Desi (Organic) Hack</h4>
#                     <p>{data.get('organic', 'Apply 10% Neem Oil solution.')}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with tab2:
#                 impact_df = pd.DataFrame({
#                     "Category": ["Pesticide Waste", "Chemical Saved"],
#                     "Liters": [affected_area_pct * field_size, chem_saved_pct * field_size]
#                 })
#                 fig = px.pie(impact_df, values="Liters", names="Category", hole=0.5,
#                              color_discrete_sequence=['#ff4b4b', '#2e7d32'])
#                 fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
#                 st.plotly_chart(fig, use_container_width=True)
#                 st.write(f"Saving **{chem_saved_pct * field_size:.1f}L** of chemicals on your {field_size} acre farm.")

#             with tab3:
#                 with st.form("expert_form"):
#                     u_phone = st.text_input("Mobile / फ़ोन")
#                     u_msg = st.text_area("Describe issue / समस्या बताएं")
#                     if st.form_submit_button(T["expert_btn"]):
#                         st.success(f"{T['expert_success']} {u_phone}")
#         st.markdown('</div>', unsafe_allow_html=True)

# # --- HISTORY SECTION ---
# st.markdown('<div class="glass-card">', unsafe_allow_html=True)
# st.subheader(T["history_header"])

# if st.session_state.scan_history:
#     df_hist = pd.DataFrame(st.session_state.scan_history)
#     h_col1, h_col2 = st.columns([2, 1])
    
#     with h_col1:
#         fig_line = px.line(df_hist, x="Time", y="Score", markers=True, color_discrete_sequence=['#90EE90'])
#         fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", yaxis_range=[0, 10])
#         st.plotly_chart(fig_line, use_container_width=True)
    
#     with h_col2:
#         st.write(f"**{T['history_insight']}:**")
#         st.warning(df_hist['Condition'].mode()[0])
#         st.dataframe(df_hist.tail(3), hide_index=True)
# else:
#     st.info("Perform a scan to see health history.")
# st.markdown('</div>', unsafe_allow_html=True)
