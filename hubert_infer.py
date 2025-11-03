import torch
import librosa
import numpy as np
import joblib
from transformers import Wav2Vec2FeatureExtractor, HubertModel

print("🔹 Loading HuBERT model and utilities...")

# ✅ Load the HuBERT model, scaler, and label encoder
model = joblib.load("hubert_model.pkl")
scaler = joblib.load("hubert_scaler.pkl")
encoder = joblib.load("hubert_label_encoder.pkl")

# ✅ Load HuBERT base model from Hugging Face
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

# 🔹 Function to extract HuBERT embeddings
def extract_hubert_features(file_path):
    speech, sr = librosa.load(file_path, sr=16000)
    inputs = extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()
    return embeddings

# 🔹 Get input file path
file_path = input("🎧 Enter the path to the .wav file: ").strip('"')

print("🎙 Extracting HuBERT features...")
features = extract_hubert_features(file_path).reshape(1, -1)

# 🔹 Scale and predict
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)
predicted_label = encoder.inverse_transform(prediction)[0]

print(f"✅ Predicted Accent: {predicted_label}")
