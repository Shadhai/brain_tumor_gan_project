from src.data_loader import load_images
from src.preprocess import preprocess_pipeline
from src.imbalance_creator import create_imbalance
import numpy as np


print("🔵 Loading dataset...")
X, y = load_images()
print("Shape X:", X.shape)
print("Shape y:", y.shape)


print("\n🟢 Preprocessing split...")
data = preprocess_pipeline(X, y)

print("Train:", data["train_images"].shape)
print("Val:", data["val_images"].shape)


print("\n🟡 Creating imbalance...")
X_imb, y_imb = create_imbalance(
    data["train_images"],
    data["train_labels"]
)

print("Imbalanced X:", X_imb.shape)

unique, counts = np.unique(y_imb, return_counts=True)
print("\nClass distribution after imbalance:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")