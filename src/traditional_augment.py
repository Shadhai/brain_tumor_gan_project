import os, numpy as np
from torchvision import transforms
from PIL import Image
from config import *
from src.data_loader import load_images
from src.imbalance_creator import create_imbalance

def create_traditional_aug():
    X, y = load_images(TRAIN_DIR)
    X_imb, y_imb = create_imbalance(X, y)

    pit_mask = y_imb == PITUITARY_IDX
    real_pit = X_imb[pit_mask]   # (200, 1, H, W)

    # Augmentations that work directly on PIL Image
    aug_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor()   # convert to tensor at the end (if needed)
    ])

    dest_dir = os.path.join(SYNTHETIC_DIR, "D5", "pituitary")
    os.makedirs(dest_dir, exist_ok=True)

    generated = 0
    for img_arr in real_pit:
        # Convert numpy array (1,H,W) to PIL Image (mode='L', grayscale)
        pil_img = Image.fromarray((img_arr[0] * 255).astype(np.uint8), mode='L')
        for _ in range(3):
            if generated >= 600:
                break
            # Apply augmentations; transforms return tensor, convert back to PIL for saving
            aug_tensor = aug_transform(pil_img)   # tensor (1, H, W)
            aug_img = transforms.ToPILImage()(aug_tensor)
            aug_img.save(os.path.join(dest_dir, f"trad_aug_{generated:04d}.png"))
            generated += 1
        if generated >= 600:
            break
    print(f"Created {generated} traditionally augmented pituitary images in D5.")

if __name__ == "__main__":
    create_traditional_aug()