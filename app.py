import streamlit as st
import numpy as np
import torch
import librosa
import joblib
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# ==============================================================
# 🌈 PAGE CONFIG & STYLING
# ==============================================================
st.set_page_config(page_title="Accent Cuisine App", page_icon="🎧", layout="centered")

# Gradient background + button styling
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #f6d365, #fda085);
        color: black;
    }

    h1, h2, h3, h4, h5, h6 {
        color: black;
        text-align: center;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #ff9966, #ff5e62);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff5e62, #ff9966);
        transform: scale(1.05);
    }

    section[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(120deg, #f6d365, #fda085);
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================
# 🎧 LOAD MODEL AND UTILITIES
# ==============================================================
@st.cache_resource
def load_models():
    model = joblib.load("hubert_model.pkl")
    scaler = joblib.load("hubert_scaler.pkl")
    encoder = joblib.load("hubert_label_encoder.pkl")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()
    return model, scaler, encoder, feature_extractor, hubert

model, scaler, encoder, feature_extractor, hubert = load_models()

# ==============================================================
# 🍴 CUISINE RECOMMENDATION MAP
# ==============================================================
cuisine_map = {
    "andhra_pradesh": ["Pulihora", "Gongura Pachadi", "Pesarattu"],
    "gujrat": ["Dhokla", "Undhiyu", "Thepla"],
    "jharkhand": ["Thekua", "Litti Chokha", "Rugra Curry"],
    "karnataka": ["Bisi Bele Bath", "Ragi Mudde", "Mysore Pak"],
    "kerala": ["Appam", "Puttu", "Avial"],
    "tamil": ["Idli", "Dosa", "Pongal"]
}

# ==============================================================
# 🧠 FUNCTION TO PREDICT ACCENT
# ==============================================================
def predict_accent(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert(**inputs)
        hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    emb = np.mean(hidden_states, axis=0).reshape(1, -1)
    X_scaled = scaler.transform(emb)
    pred = model.predict(X_scaled)
    accent = encoder.inverse_transform(pred)[0]
    return accent

# ==============================================================
# 🖥️ UI LAYOUT
# ==============================================================
st.title("🎧 Native Language Accent Cuisine Recommender")
st.subheader("Identify your accent and get regional cuisine suggestions 🍛")

uploaded_file = st.file_uploader("Upload your English speech (.wav file)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Analyze Accent"):
        with st.spinner("🎙 Analyzing accent... please wait..."):
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.read())

            accent = predict_accent("temp_audio.wav")
            st.success(f"✅ Detected Accent: {accent.upper()}")

            cuisines = cuisine_map.get(accent.lower(), ["Regional dishes not found"])
            st.subheader("🍽 Recommended Regional Cuisines:")
            for dish in cuisines:
                st.markdown(f"- {dish}")

st.markdown("---")
st.caption("Developed for the NLI Accent-Aware Cuisine Recommendation Project using HuBERT 🧠🇮🇳")
