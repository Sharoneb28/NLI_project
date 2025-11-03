# cuisine_recommendation.py
import torch
import numpy as np
import librosa
import joblib
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# ===============================
# 1️⃣ Load Model + Preprocessing
# ===============================
print("🔹 Loading HuBERT model and utilities...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

hubert_model = joblib.load("hubert_model.pkl")
hubert_scaler = joblib.load("hubert_scaler.pkl")
hubert_encoder = joblib.load("hubert_label_encoder.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# ===============================
# 2️⃣ Cuisine Mapping
# ===============================
cuisine_map = {
    "andhra_pradesh": ["Pesarattu", "Pulihora", "Gongura Pachadi"],
    "gujrat": ["Dhokla", "Thepla", "Undhiyu"],
    "jharkhand": ["Thekua", "Litti Chokha", "Rugra Curry"],
    "karnataka": ["Bisi Bele Bath", "Ragi Mudde", "Neer Dosa"],
    "kerala": ["Appam", "Puttu", "Avial", "Fish Curry"],
    "tamil": ["Idli", "Dosa", "Pongal", "Chettinad Chicken"]
}

# ===============================
# 3️⃣ Take Audio Input
# ===============================
wav_path = input("🎧 Enter path to your English speech (.wav): ").strip()

# Load and process audio
speech, sr = librosa.load(wav_path, sr=16000)
input_values = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
input_values = {k: v.to(device) for k, v in input_values.items()}

# ===============================
# 4️⃣ Extract HuBERT Features
# ===============================
print("🎙 Extracting HuBERT features...")
with torch.no_grad():
    outputs = model(**input_values)
    hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
features = np.mean(hidden_states, axis=0).reshape(1, -1)

# Scale features
features_scaled = hubert_scaler.transform(features)

# ===============================
# 5️⃣ Predict Accent
# ===============================
pred_label = hubert_model.predict(features_scaled)[0]
region = hubert_encoder.inverse_transform([pred_label])[0]
print(f"\n✅ Detected Accent: {region.upper()}")

# ===============================
# 6️⃣ Cuisine Recommendation
# ===============================
if region in cuisine_map:
    dishes = cuisine_map[region]
    print("\n🍽 Recommended Regional Cuisines:")
    for dish in dishes:
        print(f"   • {dish}")
else:
    print("⚠️ No cuisine data found for this region.")
