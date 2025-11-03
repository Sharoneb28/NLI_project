# hubert_layerwise_analysis.py
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ============================================================
print("🔹 Loading HuBERT model (facebook/hubert-base-ls960)...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# ============================================================
# Load metadata
df_meta = pd.read_csv("indic_accent_metadata.csv")
print(f"✅ Loaded metadata with {len(df_meta)} audio files")

# ============================================================
# Extract HuBERT embeddings per layer
layer_features = {i: [] for i in range(1, 13)}
labels = []

for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Extracting HuBERT layer embeddings"):
    try:
        wav_path = row["filepath"]
        label = row["state"]

        speech, sr = librosa.load(wav_path, sr=16000)
        input_values = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = {k: v.to(device) for k, v in input_values.items()}

        with torch.no_grad():
            outputs = model(**input_values)
            hidden_states = outputs.hidden_states  # Tuple of 13 tensors (1 + 12 layers)

        # Each layer: average over time → single 768-D vector
        for i in range(1, 13):  # skip layer 0 (feature extractor)
            emb = torch.mean(hidden_states[i], dim=1).squeeze(0).cpu().numpy()
            layer_features[i].append(emb)

        labels.append(label)

    except Exception as e:
        print(f"⚠️  Skipping {wav_path}: {e}")

# ============================================================
# Evaluate each layer
accuracies = []

print("\n🧠 Training + evaluating per-layer classifiers...")
le = LabelEncoder()
y_encoded = le.fit_transform(labels)

for i in range(1, 13):
    print(f"\n🔹 Evaluating Layer {i}...")
    X = np.array(layer_features[i])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"✅ Layer {i} Accuracy: {acc:.4f}")

# ============================================================
# Plot layer accuracies
plt.figure(figsize=(8,5))
plt.plot(range(1,13), accuracies, marker='o')
plt.title("HuBERT Layer-wise Accent Classification Accuracy")
plt.xlabel("HuBERT Layer")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("hubert_layerwise_accuracy.png")
plt.show()

best_layer = np.argmax(accuracies) + 1
print(f"\n🏆 Best Layer: {best_layer} with Accuracy {accuracies[best_layer-1]:.4f}")
