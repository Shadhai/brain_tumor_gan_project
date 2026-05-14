import os, cv2, numpy as np, torch
from config import IMG_SIZE, CLASSES, DEVICE

def load_images(data_dir):
    images, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path): continue
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(idx)
    X = np.array(images, dtype=np.float32)[:, np.newaxis, :, :] / 255.0   # [0,1]
    y = np.array(labels)
    return X, y

def load_single_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return img[np.newaxis, np.newaxis, :, :]   # (1,1,H,W)

def to_tensor(X, y=None):
    """Convert numpy arrays to PyTorch tensors."""
    X_t = torch.FloatTensor(X).to(DEVICE)
    if y is not None:
        y_t = torch.LongTensor(y).to(DEVICE)
        return X_t, y_t
    return X_t