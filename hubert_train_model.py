# hubert_train_model.py
import os
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import Wav2Vec2FeatureExtractor, HubertModel

print("🔹 Loading HuBERT base model (facebook/hubert-base-ls960)...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

# ==============================================
# 2️⃣ Load metadata of all .wav files
# ==============================================
df_meta = pd.read_csv("indic_accent_metadata.csv")   # you created this earlier
print(f"✅ Loaded metadata with {len(df_meta)} audio files")

# ==============================================
# 3️⃣ Extract HuBERT embeddings
# ==============================================
embeddings = []
labels = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

for i, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Extracting HuBERT embeddings"):
    try:
        wav_path = row["filepath"]
        label = row["state"]

        # Load audio at 16 kHz
        speech, sr = librosa.load(wav_path, sr=16000)
        input_values = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)


        # Move to GPU/CPU
        input_values = {k: v.to(device) for k, v in input_values.items()}

        # Forward pass through HuBERT
        with torch.no_grad():
            outputs = model(**input_values)
            hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()

        # Average over time frames → single 768-D vector
        emb = np.mean(hidden_states, axis=0)

        embeddings.append(emb)
        labels.append(label)

    except Exception as e:
        print(f"⚠️  Skipping {wav_path}: {e}")

# ==============================================
# 4️⃣ Save features
# ==============================================
print("\n💾 Saving extracted HuBERT features...")
df_features = pd.DataFrame(embeddings)
df_features["label"] = labels
df_features.to_csv("hubert_features.csv", index=False)
print(f"✅ Saved to hubert_features.csv with shape {df_features.shape}")

# ==============================================
# 5️⃣ Train classifier (Random Forest)
# ==============================================
print("\n🧠 Training RandomForest model on HuBERT embeddings...")

X = df_features.drop(columns=["label"])
y = df_features["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully! Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# ==============================================
# 6️⃣ Save model + encoders
# ==============================================
import joblib
joblib.dump(clf, "hubert_model.pkl")
joblib.dump(le, "hubert_label_encoder.pkl")
joblib.dump(scaler, "hubert_scaler.pkl")

print("\n💾 Saved HuBERT model, label encoder, and scaler!")
