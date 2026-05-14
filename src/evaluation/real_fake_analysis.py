import os, torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from config import *
from src.data_loader import load_images, load_single_image

# ------------------------------------------------------------------
# 1. Binary classifier (same as before)
# ------------------------------------------------------------------
class RealFakeCNN(nn.Module):
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

# ------------------------------------------------------------------
# 2. Feature extractor (pretrained ResNet‑18)
# ------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify first conv for 1 channel
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # remove fc

    def forward(self, x):
        return self.features(x).flatten(1)  # (batch, 512)

# ------------------------------------------------------------------
# 3. Plotting helpers
# ------------------------------------------------------------------
def plot_training_curves(train_losses, val_losses, val_accs):
    fig, ax1 = plt.subplots(figsize=(8,5))
    epochs = range(1, len(train_losses)+1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accs, 'g--', label='Val Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper right')
    plt.title('Real‑vs‑Fake Classifier Training')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'real_fake_training.png'), dpi=150)
    plt.close()
    print("Saved real_fake_training.png")

def plot_tsne(features_real, features_fake):
    """features_real and features_fake are numpy arrays of shape (N, D)"""
    n = min(len(features_real), len(features_fake))
    idx_r = np.random.choice(len(features_real), n, replace=False)
    idx_f = np.random.choice(len(features_fake), n, replace=False)
    combined = np.concatenate([features_real[idx_r], features_fake[idx_f]], axis=0)
    labels = np.array(['Real']*n + ['Fake']*n)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(combined)

    plt.figure(figsize=(8,6))
    for lbl, color in zip(['Real', 'Fake'], ['blue', 'red']):
        mask = labels == lbl
        plt.scatter(reduced[mask, 0], reduced[mask, 1], c=color, label=lbl, alpha=0.6, s=30)
    plt.legend()
    plt.title('t‑SNE of Real vs. GAN‑Generated Pituitary Images')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'real_fake_tsne.png'), dpi=150)
    plt.close()
    print("Saved real_fake_tsne.png")

def plot_sample_comparison(real_imgs, fake_imgs, n=5):
    """Show n real and n fake images side by side."""
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    idx_r = np.random.choice(len(real_imgs), n, replace=False)
    idx_f = np.random.choice(len(fake_imgs), n, replace=False)
    for i in range(n):
        axes[0, i].imshow(real_imgs[idx_r[i], 0], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Real', size=12)
        axes[1, i].imshow(fake_imgs[idx_f[i], 0], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Fake', size=12)
    plt.suptitle('Real vs. GAN‑Generated Pituitary MRI Slices')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'real_fake_samples.png'), dpi=150)
    plt.close()
    print("Saved real_fake_samples.png")

# ------------------------------------------------------------------
# 4. Main analysis
# ------------------------------------------------------------------
def run_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load real pituitary test images
    X_test, y_test = load_images(TEST_DIR)
    real_pit = X_test[y_test == PITUITARY_IDX]

    # Load synthetic pituitary from D3
    synth_dir = os.path.join(SYNTHETIC_DIR, "D3", "pituitary")
    files = [f for f in os.listdir(synth_dir) if f.endswith('.png')]
    X_synth = [load_single_image(os.path.join(synth_dir, f)) for f in files]
    X_synth = np.concatenate(X_synth, axis=0)

    # Balanced dataset
    n_min = min(len(real_pit), len(X_synth))
    idx_real = np.random.choice(len(real_pit), n_min, replace=False)
    idx_fake = np.random.choice(len(X_synth), n_min, replace=False)
    X_real_sel = real_pit[idx_real]
    X_fake_sel = X_synth[idx_fake]

    X = np.concatenate([X_real_sel, X_fake_sel], axis=0)
    y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Train / val / test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test2, y_val, y_test2 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_set = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    val_set   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val).unsqueeze(1))
    test_set  = TensorDataset(torch.FloatTensor(X_test2), torch.FloatTensor(y_test2).unsqueeze(1))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32)
    test_loader  = DataLoader(test_set,  batch_size=32)

    # Train classifier
    model = RealFakeCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses, val_accs = [], [], []
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

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:2d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc*100:.1f}%")

    # Test accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total * 100
    print(f"\nReal‑vs‑Fake Test Accuracy: {test_acc:.2f}%")

    with open(os.path.join(OUTPUT_DIR, "real_fake_accuracy.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.2f}%\n")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_accs)

    # Feature extraction for t‑SNE (use pretrained ResNet)
    feature_model = FeatureExtractor().to(device)
    feature_model.eval()

    # Extract features from the full test set (all real pituitary) and all synthetic
    def extract_features(images, batch_size=32):
        dataset = TensorDataset(torch.FloatTensor(images))
        loader = DataLoader(dataset, batch_size=batch_size)
        feats = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                f = feature_model(batch)
                feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    # Use all test real pituitary and a subset of synthetic to not overload t‑SNE
    real_feats = extract_features(X_test[y_test == PITUITARY_IDX])
    fake_feats = extract_features(X_synth[:len(real_feats)])  # match size

    plot_tsne(real_feats, fake_feats)

    # Sample image comparison
    plot_sample_comparison(real_pit, X_synth, n=5)

if __name__ == "__main__":
    os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
    run_analysis()