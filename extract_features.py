import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# Load metadata
df = pd.read_csv("indic_accent_metadata.csv")

features = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔊 Extracting features"):
    file_path = row["filepath"]
    label = row["state"]

    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        features.append([label] + mfcc_mean.tolist())
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# Save as CSV
columns = ["label"] + [f"mfcc_{i}" for i in range(13)]
feature_df = pd.DataFrame(features, columns=columns)
feature_df.to_csv("accent_features.csv", index=False)

print("\n✅ Feature extraction complete! Saved as accent_features.csv")
print(f"Total processed: {len(feature_df)} rows")
