import torch, numpy as np, matplotlib.pyplot as plt

def generate_samples(generator, latent_dim, n=25, device='cuda'):
    generator.eval()
    noise = torch.randn(n, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        imgs = generator(noise).cpu()
    imgs = (imgs + 1) / 2.0
    return imgs

def plot_generated_grid(imgs, save_path=None):
    """imgs: tensor (25,1,H,W)"""
    fig, axs = plt.subplots(5,5, figsize=(10,10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(imgs[i,0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()