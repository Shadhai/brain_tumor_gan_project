import os, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from config import *
from src.data_loader import load_images, load_single_image

class RealFakeCNN(nn.Module):
    """
    Small CNN for binary real‑vs‑fake classification.
    It takes a 128×128 grayscale image and outputs a probability (real = 1).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def evaluate_real_fake():
    # -------------------------------
    # 1. Load real pituitary test images
    # -------------------------------
    X_test, y_test = load_images(TEST_DIR)
    real_pit = X_test[y_test == PITUITARY_IDX]   # (351, 1, 128, 128)

    # -------------------------------
    # 2. Load synthetic pituitary images (from D3)
    # -------------------------------
    synth_dir = os.path.join(SYNTHETIC_DIR, "D3", "pituitary")
    files = [f for f in os.listdir(synth_dir) if f.endswith('.png')]
    X_synth = [load_single_image(os.path.join(synth_dir, f)) for f in files]
    X_synth = np.concatenate(X_synth, axis=0)     # (600, 1, 128, 128)

    # -------------------------------
    # 3. Create a balanced dataset (equal number of real and fake)
    # -------------------------------
    n_min = min(len(real_pit), len(X_synth))      # 351
    idx_real = np.random.choice(len(real_pit), n_min, replace=False)
    idx_fake = np.random.choice(len(X_synth), n_min, replace=False)
    X_real_sel = real_pit[idx_real]
    X_fake_sel = X_synth[idx_fake]

    X = np.concatenate([X_real_sel, X_fake_sel], axis=0)
    y = np.concatenate([np.ones(n_min), np.zeros(n_min)])   # 1 = real, 0 = fake

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # -------------------------------
    # 4. Train / validation / test split (80 / 10 / 10)
    # -------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test2, y_val, y_test2 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_set = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    val_set   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val).unsqueeze(1))
    test_set  = TensorDataset(torch.FloatTensor(X_test2), torch.FloatTensor(y_test2).unsqueeze(1))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32)
    test_loader  = DataLoader(test_set,  batch_size=32)

    # -------------------------------
    # 5. Model, loss, optimizer
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealFakeCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------------
    # 6. Training loop
    # -------------------------------
    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc*100:.1f}%")

    # -------------------------------
    # 7. Final test accuracy
    # -------------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total * 100
    print(f"\n=== Real‑vs‑Fake Classifier Test Accuracy: {test_acc:.2f}% ===")

    # Save result
    with open(os.path.join(OUTPUT_DIR, "real_fake_accuracy.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.2f}%\n")
    print("Result saved to outputs/real_fake_accuracy.txt")

if __name__ == "__main__":
    evaluate_real_fake()