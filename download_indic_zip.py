from huggingface_hub import hf_hub_download
import zipfile
import os

repo_id = "DarshanaS/IndicAccentDb"
filename = "IndicAccentDB.zip"

print("📦 Downloading dataset ZIP from Hugging Face...")
zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

print(f"✅ Downloaded ZIP to: {zip_path}")

# Extract the ZIP to a local folder (in your project)
extract_folder = "indic_dataset"
os.makedirs(extract_folder, exist_ok=True)

print(f"📂 Extracting files to '{extract_folder}'...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_folder)

print("✅ Extraction complete!")
print(f"Contents of {extract_folder}:")
print(os.listdir(extract_folder))
