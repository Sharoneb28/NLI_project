import pandas as pd

# Load the metadata CSV
df = pd.read_csv("indic_accent_metadata.csv")

print("✅ Loaded metadata!")
print(f"Total entries: {len(df)}")
print("\nUnique states:", df['state'].unique())

# Show first few rows
print("\nSample entries:\n", df.head())
