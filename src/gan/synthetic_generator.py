import numpy as np
import os
from PIL import Image


def generate_and_save(generator, latent_dim, num_images, save_path):

    os.makedirs(save_path, exist_ok=True)

    noise = np.random.normal(0, 1, (num_images, latent_dim))
    gen_images = generator.predict(noise, verbose=0)

    for i, img in enumerate(gen_images):

        img = (img * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img.squeeze()).save(
            os.path.join(save_path, f"synthetic_{i}.png")
        )