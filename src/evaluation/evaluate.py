import os, json, torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from config import *
from src.data_loader import load_images
from src.preprocess import preprocess_for_classifier
from src.model.classifier import build_classifier

def evaluate_on_test():
    X_test, y_test = load_images(TEST_DIR)
    X_test = preprocess_for_classifier(X_test)
    test_set = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    results = {}
    for exp in SYNTHETIC_COUNTS:
        # Load best model (pick one with highest accuracy from validation – we use first seed for simplicity)
        model_path = os.path.join(MODEL_DIR, f"classifier_{exp}_seed42.pth")
        if not os.path.exists(model_path):
            continue
        model = build_classifier().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(lbls.numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        cm = confusion_matrix(all_targets, all_preds)
        np.save(os.path.join(OUTPUT_DIR, "confusion_matrices", f"{exp}_cm.npy"), cm)

        results[exp] = {"accuracy": acc, "f1": f1}
        print(f"{exp} Test Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(classification_report(all_targets, all_preds, target_names=CLASSES))

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    evaluate_on_test()