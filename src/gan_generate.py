import os, torch, numpy as np
from PIL import Image
from config import *
from src.gan.generator import build_generator

def generate_synthetic():
    generator = build_generator(LATENT_DIM).to(DEVICE)
    state_dict = torch.load(os.path.join(MODEL_DIR, "gan_generator.pth"), map_location=DEVICE)
    generator.load_state_dict(state_dict)
    generator.eval()

    for exp, count in SYNTHETIC_COUNTS.items():
        if count == 0:
            continue
        dest = os.path.join(SYNTHETIC_DIR, exp, "pituitary")
        os.makedirs(dest, exist_ok=True)
        noise = torch.randn(count, LATENT_DIM, 1, 1, device=DEVICE)
        with torch.no_grad():
            imgs = generator(noise)
        imgs = (imgs * 0.5 + 0.5) * 255
        imgs = imgs.cpu().squeeze(1).numpy().astype(np.uint8)  # (count, H, W)
        for i in range(count):
            img = Image.fromarray(imgs[i], mode='L')
            img.save(os.path.join(dest, f"synthetic_{i:04d}.png"))
        print(f"Saved {count} synthetic pituitary images for {exp}")

if __name__ == "__main__":
    generate_synthetic()