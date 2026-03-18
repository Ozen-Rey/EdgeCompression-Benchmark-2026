"""
Selettore baseline statico per l'ablation study.
Non usa feature dell'immagine — predice il codec più frequente
per fascia di bpp nel dataset di training.
Confrontato con XGBoost e CNN sullo stesso hold-out set.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.expanduser("~/tesi"))

ORACLE_CSV = os.path.expanduser("~/tesi/results/oracle/oracle_v2_multi.csv")
RESULTS_DIR = os.path.expanduser("~/tesi/results/selector")


def train_baseline(df: pd.DataFrame) -> tuple[float, float]:
    """
    Selettore statico: per ogni immagine di test predice il codec
    più frequente nel training set (majority class baseline).
    Questo è il lower bound — qualsiasi modello serio deve batterlo.
    """
    y_raw = np.array(df["winner_codec"].values)

    le = LabelEncoder()
    # Forziamo np.array per placare Pylance (risolve ArrayLike issue)
    y = np.array(le.fit_transform(y_raw))
    classes = np.array(le.classes_)

    print(f"\nClassi: {classes}")
    print(f"Distribuzione:\n{pd.Series(y_raw).value_counts()}")

    # split 80/20 — stesso random_state di XGBoost e CNN
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )
    idx_train = np.array(idx_train)
    idx_test = np.array(idx_test)

    y_train = y[idx_train]
    y_test = y[idx_test]

    print(f"\nTrain: {len(y_train)} immagini, Test: {len(y_test)} immagini")

    # strategia 1: majority class — predice sempre la classe più frequente
    majority_class = int(np.bincount(y_train).argmax())
    y_pred_majority = np.full_like(y_test, majority_class)
    acc_majority = float((y_pred_majority == y_test).mean())

    print(f"\n--- Baseline Majority Class ---")
    print(f"Classe predetta sempre: {classes[majority_class]}")
    print(f"Accuracy: {acc_majority:.3f} ({acc_majority * 100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_majority, target_names=classes))

    # strategia 2: distribuzione proporzionale — campiona dalla distribuzione del training
    train_probs = np.bincount(y_train) / len(y_train)
    np.random.seed(42)
    y_pred_random = np.random.choice(len(classes), size=len(y_test), p=train_probs)
    acc_random = float((y_pred_random == y_test).mean())

    print(f"\n--- Baseline Random Proportional ---")
    print(f"Probabilità per classe: {dict(zip(classes, train_probs.round(3)))}")
    print(f"Accuracy: {acc_random:.3f} ({acc_random * 100:.1f}%)")

    # plot confusion matrix majority
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(y_test, y_pred_majority)
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_xticklabels(classes, rotation=45)
    axes[0].set_yticklabels(classes)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Baseline Majority\nAccuracy: {acc_majority:.1%}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[0].text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar(im, ax=axes[0])

    # distribuzione classi nel dataset
    class_counts = pd.Series(y_raw).value_counts()
    axes[1].bar(class_counts.index, class_counts.values, color="steelblue")
    axes[1].set_title("Distribuzione vincitori nel dataset")
    axes[1].set_ylabel("Numero immagini")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "baseline_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot salvato in {plot_path}")

    return acc_majority, acc_random


def print_ablation_summary(
    acc_baseline: float,
    acc_xgboost: float | None = None,
    acc_cnn: float | None = None,
) -> None:
    """Stampa la tabella riassuntiva dell'ablation study."""
    print(f"\n{'=' * 60}")
    print("ABLATION STUDY — CONFRONTO SELETTORI")
    print(f"{'=' * 60}")
    print(f"{'Selettore':<25} {'Accuracy':>10} {'Latenza':>12} {'Interpretabile':>15}")
    print(f"{'-' * 60}")
    print(f"{'Baseline (majority)':<25} {acc_baseline:>10.1%} {'<1ms':>12} {'Sì':>15}")
    if acc_xgboost is not None:
        delta = acc_xgboost - acc_baseline
        print(
            f"{'XGBoost (8 feature)':<25} {acc_xgboost:>10.1%} "
            f"{'~40ms':>12} {'Sì':>15}  (+{delta:.1%})"
        )
    if acc_cnn is not None:
        delta = acc_cnn - acc_baseline
        print(
            f"{'Micro-CNN (128x128)':<25} {acc_cnn:>10.1%} "
            f"{'~5ms':>12} {'No':>15}  (+{delta:.1%})"
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    print("Caricamento dati oracolo...")
    df = pd.read_csv(ORACLE_CSV)
    df = df[df["winner_codec"] != "ERROR"].reset_index(drop=True)
    print(f"Immagini valide: {len(df)}")

    acc_majority, acc_random = train_baseline(df)

    # prova a caricare risultati XGBoost e CNN se disponibili
    xgboost_acc = None
    cnn_acc = None

    xgboost_model_path = os.path.join(RESULTS_DIR, "xgboost_model.pkl")
    if os.path.exists(xgboost_model_path):
        import joblib
        from src.selector.feature_extractor import SpatialFeatureExtractor
        from sklearn.preprocessing import LabelEncoder as LE

        saved = joblib.load(xgboost_model_path)
        model = saved["model"]
        le = saved["label_encoder"]
        feature_cols = saved["features"]

        extractor = SpatialFeatureExtractor()
        y_raw = np.array(df["winner_codec"].values)
        le2 = LE()
        
        # Risolve l'errore di len(y) forzando np.array
        y = np.array(le2.fit_transform(y_raw))
        idx = np.arange(len(y))
        _, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

        features = []
        valid_idx = []
        for i in idx_test:
            try:
                feat = extractor.extract_features(str(df.iloc[i]["image"]))
                features.append([feat[c] for c in feature_cols])
                valid_idx.append(i)
            except Exception:
                continue

        if features:
            X_test = np.array(features)
            valid_idx_arr = np.array(valid_idx)
            
            # Diviso in passaggi espliciti per evitare l'errore di slicing di Pylance
            y_subset = y[valid_idx_arr]
            inverse_y = le2.inverse_transform(y_subset)
            y_test = np.array(le.transform(inverse_y))
            
            y_pred = model.predict(X_test)
            xgboost_acc = float((y_pred == y_test).mean())
            print(f"\nXGBoost accuracy (dal file salvato): {xgboost_acc:.1%}")

    print_ablation_summary(acc_majority, xgboost_acc, cnn_acc)

    print(f"\n{'=' * 60}")
    print(f"BASELINE FINALE: {acc_majority:.1%} (majority class)")
    print(f"{'=' * 60}")