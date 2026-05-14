import os, json, torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from scipy.linalg import sqrtm
from config import *
from src.data_loader import load_images, load_single_image

# ------------------------------------------------------------------
# 1. FEATURE EXTRACTION (BATCHED)
# ------------------------------------------------------------------
def get_inception_features(images, model, batch_size=32):
    """
    images: numpy array (N, 1, H, W) in [0,1]
    Returns: numpy array of shape (N, 2048)
    """
    # Convert to 3‑channel RGB
    imgs_rgb = np.repeat(images, 3, axis=1)          # (N, 3, H, W)
    imgs_rgb = torch.FloatTensor(imgs_rgb)            # (N, 3, H, W)

    # Resize to 299x299
    up = transforms.Resize((299, 299))
    imgs_rgb = up(imgs_rgb)                           # (N, 3, 299, 299)

    # Normalize to [-1, 1] (Inception expects)
    imgs_rgb = (imgs_rgb - 0.5) / 0.5

    dataset = TensorDataset(imgs_rgb)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features_list = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            feats = model(batch)                     # (batch, 2048)
            features_list.append(feats.cpu().numpy())

    return np.concatenate(features_list, axis=0)

# ------------------------------------------------------------------
# 2. FID CALCULATION
# ------------------------------------------------------------------
def calculate_fid(model, real_imgs, fake_imgs, batch_size=16):
    act_real = get_inception_features(real_imgs, model, batch_size=batch_size)
    act_fake = get_inception_features(fake_imgs, model, batch_size=batch_size)

    mu1, sigma1 = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu2, sigma2 = act_fake.mean(axis=0), np.cov(act_fake, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# ------------------------------------------------------------------
# 3. MAIN – COMPUTE FID FOR ALL EXPERIMENTS
# ------------------------------------------------------------------
def compute_fid_for_experiments():
    # Free GPU memory left over from previous steps
    torch.cuda.empty_cache()

    # Load Inception v3 as feature extractor
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.fc = torch.nn.Identity()       # remove classification head
    model.eval()
    model = model.to(DEVICE)

    # Load real pituitary test images
    X_test, y_test = load_images(TEST_DIR)
    real_pit = X_test[y_test == PITUITARY_IDX]   # (351, 1, H, W)

    fids = {}
    for exp, count in SYNTHETIC_COUNTS.items():
        if count == 0:
            continue
        synth_dir = os.path.join(SYNTHETIC_DIR, exp, "pituitary")
        files = [f for f in os.listdir(synth_dir) if f.endswith('.png')]
        X_synth = [load_single_image(os.path.join(synth_dir, f)) for f in files]
        X_synth = np.concatenate(X_synth, axis=0)

        # Use smaller batch size to avoid OOM
        fid = calculate_fid(model, real_pit, X_synth, batch_size=8)
        fids[exp] = round(fid, 2)
        print(f"FID for {exp}: {fid:.2f}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "fid_scores.json"), 'w') as f:
        json.dump(fids, f, indent=2)
    print("FID scores saved.")

if __name__ == "__main__":
    compute_fid_for_experiments()