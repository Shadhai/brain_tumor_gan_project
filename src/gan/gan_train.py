import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from config import *
from src.data_loader import load_images
from src.preprocess import preprocess_for_gan
from src.gan.generator import build_generator
from src.gan.discriminator import build_discriminator

def train_gan():
    # Load only pituitary images
    X, y = load_images(TRAIN_DIR)
    mask = y == PITUITARY_IDX
    X_pit = X[mask]
    X_pit = preprocess_for_gan(X_pit)   # -> [-1, 1]
    dataset = TensorDataset(torch.FloatTensor(X_pit))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    generator = build_generator(LATENT_DIM).to(DEVICE)
    discriminator = build_discriminator().to(DEVICE)

    g_optim = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(25, LATENT_DIM, 1, 1, device=DEVICE)
    sample_dir = os.path.join(OUTPUT_DIR, "gan_samples")
    os.makedirs(sample_dir, exist_ok=True)

    print("Starting GAN training on pituitary class only...")
    for epoch in range(EPOCHS_GAN):
        for i, (real_imgs,) in enumerate(loader):
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # --- Train Discriminator ---
            d_optim.zero_grad()
            # Real
            real_labels = torch.ones(batch_size, 1, device=DEVICE) * 0.9
            output_real = discriminator(real_imgs)
            d_loss_real = criterion(output_real, real_labels)

            # Fake
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = generator(noise)
            fake_labels = torch.ones(batch_size, 1, device=DEVICE) * 0.1
            output_fake = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(output_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optim.step()

            # --- Train Generator ---
            g_optim.zero_grad()
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            gen_imgs = generator(noise)
            output = discriminator(gen_imgs)
            g_loss = criterion(output, torch.ones(batch_size, 1, device=DEVICE))
            g_loss.backward()
            g_optim.step()

        if epoch % 100 == 0:
            d_acc = (output_real > 0.5).float().mean().item() * 100
            print(f"Epoch {epoch:4d} | D loss {d_loss.item():.4f} | D acc {d_acc:.1f}% | G loss {g_loss.item():.4f}")

            # Save sample images
            with torch.no_grad():
                gen_samples = generator(fixed_noise).cpu()
            gen_samples = (gen_samples + 1) / 2.0   # to [0,1]
            fig, axs = plt.subplots(5, 5, figsize=(10,10))
            for j, ax in enumerate(axs.flat):
                ax.imshow(gen_samples[j, 0], cmap='gray')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f'epoch_{epoch}.png'))
            plt.close()

        if epoch > 0 and epoch % 500 == 0:
            torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f'generator_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, f'discriminator_{epoch}.pth'))

    torch.save(generator.state_dict(), os.path.join(MODEL_DIR, 'gan_generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, 'gan_discriminator.pth'))
    print("GAN training complete.")

if __name__ == "__main__":
    train_gan()