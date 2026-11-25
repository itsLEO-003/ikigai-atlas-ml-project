
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "best_career_model.pkl")
compressed_path = os.path.join(BASE_DIR, "best_career_model_compressed.pkl")

model = joblib.load(model_path)

joblib.dump(
    model,
    compressed_path,
    compress=3
)

original = os.path.getsize(model_path)
compressed = os.path.getsize(compressed_path)

print(f"Original size: {original / (1024 * 1024):.2f} MB")
print(f"Compressed size: {compressed / (1024 * 1024):.2f} MB")
