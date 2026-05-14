import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_for_gan(X):
    """Normalise to [-1, 1]."""
    return X * 2.0 - 1.0

def preprocess_for_classifier(X):
    """Already [0,1]."""
    return X

def create_train_val_stratified(X, y, seed=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return X_train, X_val, y_train, y_val