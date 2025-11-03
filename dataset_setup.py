# dataset_setup.py
from datasets import load_dataset
import pandas as pd

print("🔹 Loading IndicAccentDb metadata via streaming (no audio decoding)...")

# Stream safely without decoding audio
streamed_dataset = load_dataset("DarshanaS/IndicAccentDb", split="train", streaming=True)
streamed_dataset = streamed_dataset.with_format(None)

# Collect only text + labels
samples = []
for i, sample in enumerate(streamed_dataset):
    samples.append({"label": sample.get("label", "NA")})
    if i >= 50:  # take only first 50 for preview
        break

df = pd.DataFrame(samples)
df.to_csv("train_data_preview.csv", index=False)

print("✅ Saved train_data_preview.csv successfully (no audio needed).")
