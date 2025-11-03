import os
import joblib
import numpy as np
import librosa

# =========================
# Load trained model + tools
# =========================
model = joblib.load("accent_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =========================
# Function: extract MFCCs from audio
# =========================
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.reshape(1, -1)

# =========================
# Function: predict accent
# =========================
def predict_accent(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    features = extract_mfcc(file_path)
    scaled = scaler.transform(features)
    pred = model.predict(scaled)
    accent = label_encoder.inverse_transform(pred)[0]
    print(f"🎙️ Predicted Accent: {accent}")

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    test_audio = input("Enter path to a .wav file: ").strip()
    predict_accent(test_audio)
