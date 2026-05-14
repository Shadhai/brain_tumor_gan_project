import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_prob=None, num_classes=4):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred).tolist()
    result = {"accuracy": acc, "f1": f1, "recall": recall, "confusion_matrix": cm}
    if y_prob is not None:
        y_true_oh = np.eye(num_classes)[y_true]
        result["auc"] = roc_auc_score(y_true_oh, y_prob, multi_class='ovr')
    return result