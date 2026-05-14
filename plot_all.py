import os, json, numpy as np, matplotlib.pyplot as plt, torch
from torchvision import transforms
from config import *
from src.data_loader import load_images, load_single_image
from src.model.classifier import build_classifier
from src.explainability.gradcam import make_gradcam_heatmap  # you already have this

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
def load_summaries():
    data = {}
    for exp in SYNTHETIC_COUNTS:
        path = os.path.join(EXPERIMENT_DIR, f"{exp}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                data[exp] = json.load(f)
    return data

def parse_mean_std(val):
    """Convert '0.85 ± 0.01' to (0.85, 0.01)."""
    if isinstance(val, str) and '±' in val:
        mean, std = val.split('±')
        return float(mean.strip()), float(std.strip())
    return float(val), 0.0

# ------------------------------------------------------------------
# 2. PERFORMANCE PLOT (dual axis)
# ------------------------------------------------------------------
def plot_performance(summaries):
    x_labels = ['D1\n0', 'D2\n200', 'D3\n600', 'D4\n1000', 'D5\nTrad 600']
    acc, acc_err = [], []
    rec, rec_err = [], []

    for exp in ['D1','D2','D3','D4','D5']:
        s = summaries.get(exp, {})
        m, e = parse_mean_std(s.get('accuracy', '0'))
        acc.append(m*100); acc_err.append(e*100)
        m, e = parse_mean_std(s.get('pituitary_recall', '0'))
        rec.append(m*100); rec_err.append(e*100)

    x = np.arange(len(x_labels))
    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.errorbar(x, acc, yerr=acc_err, fmt='o-', color='tab:blue', capsize=5, label='Accuracy (%)')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)

    ax2 = ax1.twinx()
    ax2.errorbar(x, rec, yerr=rec_err, fmt='s--', color='tab:red', capsize=5, label='Pituitary Recall (%)')
    ax2.set_ylabel('Pituitary Recall (%)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='center right')

    plt.title('Classifier Performance vs Augmentation Ratio', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'accuracy_recall_vs_ratio.png'), dpi=150)
    plt.close()
    print("Saved accuracy_recall_vs_ratio.png")

# ------------------------------------------------------------------
# 3. FID PLOT
# ------------------------------------------------------------------
def plot_fid():
    fid_path = os.path.join(OUTPUT_DIR, 'fid_scores.json')
    if not os.path.exists(fid_path):
        print("FID scores not found.")
        return
    with open(fid_path) as f:
        fid_dict = json.load(f)

    # GAN points: D2, D3, D4
    x_gan = [200, 600, 1000]
    y_gan = [fid_dict.get('D2',0), fid_dict.get('D3',0), fid_dict.get('D4',0)]
    y_d5 = fid_dict.get('D5', 0)

    plt.figure(figsize=(8,5))
    plt.plot(x_gan, y_gan, 'o-', color='green', label='GAN generated')
    plt.axhline(y=y_d5, color='orange', linestyle='--', label=f'Traditional aug (FID={y_d5:.1f})')
    plt.xlabel('Number of synthetic images')
    plt.ylabel('FID (lower is better)')
    plt.title('Image Quality vs GAN Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'fid_vs_ratio.png'), dpi=150)
    plt.close()
    print("Saved fid_vs_ratio.png")

# ------------------------------------------------------------------
# 4. CONFUSION MATRICES
# ------------------------------------------------------------------
def plot_confusion_matrices():
    cm_dir = os.path.join(OUTPUT_DIR, 'confusion_matrices')
    if not os.path.exists(cm_dir):
        return
    from sklearn.metrics import ConfusionMatrixDisplay
    class_names = CLASSES
    for exp in ['D1','D3','D4','D5']:
        cm_file = os.path.join(cm_dir, f"{exp}_cm.npy")
        if os.path.exists(cm_file):
            cm = np.load(cm_file)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(6,5))
            disp.plot(cmap='Blues', ax=ax, colorbar=True)
            plt.title(f'Confusion Matrix – {exp}')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'plots', f'cm_{exp}.png'), dpi=150)
            plt.close()
            print(f"Saved cm_{exp}.png")

# ------------------------------------------------------------------
# 5. GRAD‑CAM (automatic)
# ------------------------------------------------------------------
def generate_gradcam():
    # Load test set pituitary images
    X_test, y_test = load_images(TEST_DIR)
    pit_idx = np.where(y_test == PITUITARY_IDX)[0]
    if len(pit_idx) < 5:
        print("Not enough pituitary test images for Grad‑CAM.")
        return
    test_images = X_test[pit_idx[:5]]   # pick 5 real pituitary
    test_tensors = torch.FloatTensor(test_images).to(DEVICE)

    # For each experiment, load the best model (we use first seed saved)
    for exp in ['D1','D3','D4','D5']:
        model_path = os.path.join(MODEL_DIR, f"classifier_{exp}_seed42.pth")
        if not os.path.exists(model_path):
            continue
        model = build_classifier().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        # Use last convolutional layer as target (EfficientNet features[-1] is a Conv2d)
        target_layer = model.features[-1][0]   # might need to check exact layer name
        fig, axs = plt.subplots(2, 5, figsize=(15,6))
        for i in range(5):
            img_tensor = test_tensors[i:i+1]
            # Original image
            axs[0,i].imshow(test_images[i,0], cmap='gray')
            axs[0,i].axis('off')
            # Heatmap
            heatmap = make_gradcam_heatmap(model, img_tensor, target_layer)
            axs[1,i].imshow(test_images[i,0], cmap='gray')
            axs[1,i].imshow(heatmap, cmap='jet', alpha=0.4)
            axs[1,i].axis('off')
        plt.suptitle(f'Grad‑CAM – {exp}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'plots', f'gradcam_{exp}.png'), dpi=150)
        plt.close()
        print(f"Saved gradcam_{exp}.png")

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    plot_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    summaries = load_summaries()
    if not summaries:
        print("❌ No experiment summaries found. Run classifier_train.py first.")
    else:
        plot_performance(summaries)
        plot_fid()
        plot_confusion_matrices()
        generate_gradcam()
        print("\n🎉 All plots saved in outputs/plots/")
        print("You can now copy them into your conference paper.")