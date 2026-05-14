import torch, numpy as np, matplotlib.pyplot as plt
from torch.nn import functional as F

def make_gradcam_heatmap(model, img_tensor, target_layer):
    """
    img_tensor: (1,1,H,W) tensor
    target_layer: layer object e.g. model.features[-1]
    """
    model.eval()
    gradients = []
    activations = None

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        output.register_hook(save_gradient)

    hook = target_layer.register_forward_hook(forward_hook)
    output = model(img_tensor)
    pred_class = output.argmax(dim=1)
    loss = output[0, pred_class]
    model.zero_grad()
    loss.backward()
    hook.remove()

    # Pool gradients
    pooled_gradients = torch.mean(gradients[0], dim=[0,2,3])
    # Weight activations
    heatmap = torch.mean(activations * pooled_gradients[None, :, None, None], dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    return heatmap.detach().cpu().numpy()

def save_gradcam_grid(model, images, target_layer, exp_name, save_path):
    """images: list of (1,1,H,W) tensors (5 samples)"""
    fig, axs = plt.subplots(2, 5, figsize=(15,6))
    for i in range(5):
        img = images[i]
        axs[0,i].imshow(img[0,0].cpu(), cmap='gray')
        axs[0,i].axis('off')
        heatmap = make_gradcam_heatmap(model, img.to(DEVICE), target_layer)
        axs[1,i].imshow(img[0,0].cpu(), cmap='gray')
        axs[1,i].imshow(heatmap, cmap='jet', alpha=0.4)
        axs[1,i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()