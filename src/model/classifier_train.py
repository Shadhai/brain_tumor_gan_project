import os, numpy as np, json, torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from scipy.stats import ttest_rel
from config import *
from src.data_loader import load_images, load_single_image
from src.imbalance_creator import create_imbalance
from src.preprocess import preprocess_for_classifier, create_train_val_stratified
from src.model.classifier import build_classifier

def build_dataset(exp_name):
    X, y = load_images(TRAIN_DIR)
    X, y = create_imbalance(X, y)
    if exp_name != "D1":
        synth_dir = os.path.join(SYNTHETIC_DIR, exp_name, "pituitary")
        files = [f for f in os.listdir(synth_dir) if f.endswith('.png')]
        X_synth = [load_single_image(os.path.join(synth_dir, f)) for f in files]
        X_synth = np.concatenate(X_synth, axis=0)
        y_synth = np.full(len(X_synth), PITUITARY_IDX)
        X = np.concatenate([X, X_synth], axis=0)
        y = np.concatenate([y, y_synth], axis=0)
    return preprocess_for_classifier(X), y

def run_experiment(exp_name):
    X, y = build_dataset(exp_name)
    # Store per‑seed results
    results = {'accuracy': [], 'f1': [], 'auc': [], 'pituitary_recall': []}
    best_acc = 0.0
    best_model_state = None

    for seed in SEEDS:
        X_tr, X_val, y_tr, y_val = create_train_val_stratified(X, y, seed=seed)
        train_set = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        val_set = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

        model = build_classifier().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR_CLASSIFIER)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_val_loss = float('inf')
        for epoch in range(EPOCHS_CLASSIFIER):
            model.train()
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            all_preds, all_probs, all_targets = [], [], []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(DEVICE)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_targets.extend(lbls.numpy())

            val_acc = accuracy_score(all_targets, all_preds)
            val_f1 = f1_score(all_targets, all_preds, average='weighted')
            # AUC requires probabilities
            all_targets_oh = np.eye(NUM_CLASSES)[all_targets]
            val_auc = roc_auc_score(all_targets_oh, all_probs, multi_class='ovr')
            val_recall = recall_score(all_targets, all_preds, labels=[PITUITARY_IDX], average=None)[0]

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = model.state_dict()

        # Save best model for this seed
        torch.save(best_model_state, os.path.join(MODEL_DIR, f"classifier_{exp_name}_seed{seed}.pth"))

        results['accuracy'].append(val_acc)
        results['f1'].append(val_f1)
        results['auc'].append(val_auc)
        results['pituitary_recall'].append(val_recall)

    # Save raw values
    raw_path = os.path.join(EXPERIMENT_DIR, f"{exp_name}_raw_values.json")
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary with mean ± std
    summary = {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in results.items()}
    summary['best_accuracy'] = best_acc
    summary_path = os.path.join(EXPERIMENT_DIR, f"{exp_name}_results.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"{exp_name}: {summary}")
    return results

def compute_statistical_tests(all_raw):
    comparisons = [("D1", "D3"), ("D3", "D4"), ("D3", "D5")]
    stats = {}
    for metric in ['accuracy', 'f1', 'auc', 'pituitary_recall']:
        stats[metric] = {}
        for a, b in comparisons:
            if a in all_raw and b in all_raw:
                t_stat, p_val = ttest_rel(all_raw[a][metric], all_raw[b][metric])
                stats[metric][f"{a}_vs_{b}"] = {"t_statistic": float(t_stat), "p_value": float(p_val)}
    return stats

if __name__ == "__main__":
    all_raw = {}
    for exp in SYNTHETIC_COUNTS:
        print(f"Training {exp}...")
        raw = run_experiment(exp)
        all_raw[exp] = raw

    stats = compute_statistical_tests(all_raw)
    with open(os.path.join(OUTPUT_DIR, "statistical_tests.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    print("Statistical tests saved.")