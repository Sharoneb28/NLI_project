import pickle

files = ["accent_model.pkl", "scaler.pkl", "label_encoder.pkl"]

for f in files:
    try:
        with open(f, "rb") as file:
            obj = pickle.load(file)
        print(f"✅ {f} loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading {f}: {e}")
