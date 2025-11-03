import os
import pandas as pd

# Base folder containing state subfolders
base_dir = "indic_dataset"

data = []

# Loop through each state folder
for state in os.listdir(base_dir):
    state_path = os.path.join(base_dir, state)
    if os.path.isdir(state_path):
        for filename in os.listdir(state_path):
            if filename.endswith(".wav"):
                data.append({
                    "state": state,
                    "filename": filename,
                    "filepath": os.path.join(state_path, filename)
                })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
csv_path = "indic_accent_metadata.csv"
df.to_csv(csv_path, index=False)

print(f"✅ Metadata CSV created: {csv_path}")
print(f"Total audio files listed: {len(df)}")
print("\nFirst few entries:\n", df.head())

