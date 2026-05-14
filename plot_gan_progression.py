import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from config import OUTPUT_DIR

epochs = [0, 500, 1000, 1500, 1900]
folder = os.path.join(OUTPUT_DIR, 'gan_samples')

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, ep in enumerate(epochs):
    img = mpimg.imread(os.path.join(folder, f'epoch_{ep}.png'))
    axes[i].imshow(img)
    axes[i].set_title(f'Epoch {ep}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'gan_progression.png'), dpi=150)
plt.close()
print('gan_progression.png saved.')