from huggingface_hub import HfApi

# Initialize the Hugging Face API
api = HfApi()

# The dataset we want to explore (NOT a model)
repo_id = "DarshanaS/IndicAccentDb"

print("📂 Listing dataset repo files (this may take a few seconds)...")

# Tell the API we're looking inside a dataset repo
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

# Show some info about the files found
print(f"✅ Found {len(files)} files in the dataset repository.\n")

# Print the first few to inspect
print("First few files:")
for f in files[:15]:
    print(" -", f)
