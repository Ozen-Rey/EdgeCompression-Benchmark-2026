"""
Training del selettore XGBoost sul dataset dell'oracolo.
Input: oracle_v2_multi.csv + feature estratte da feature_extractor.py
Output: modello salvato + report accuracy + feature importance
"""

import os
import sys
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.expanduser("~/tesi"))
from src.selector.feature_extractor import SpatialFeatureExtractor

ORACLE_CSV = os.path.expanduser("~/tesi/results/oracle/oracle_v2_multi.csv")
RESULTS_DIR = os.path.expanduser("~/tesi/results/selector")
MODEL_PATH = os.path.expanduser("~/tesi/results/selector/xgboost_model.pkl")
FEATURES_CACHE = os.path.expanduser("~/tesi/results/selector/features_cache.csv")

FEATURE_COLS = [
    "variance",
    "entropy",
    "sobel_mean",
    "rms_contrast",
    "laplacian_var",
    "skewness",
    "kurtosis",
    "edge_density",
]


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Estrae le 8 feature spaziali per ogni immagine nel dataset."""
    extractor = SpatialFeatureExtractor()

    # carica cache se esiste
    if os.path.exists(FEATURES_CACHE):
        cache = pd.read_csv(FEATURES_CACHE, index_col="image")
        print(f"Cache feature trovata: {len(cache)} immagini")
    else:
        cache = pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        img_path = str(row["image"])
        feat: dict[str, Any] = {}

        if len(cache) > 0 and img_path in cache.index:
            feat = {str(k): v for k, v in cache.loc[img_path].to_dict().items()}
        else:
            try:
                feat = cast(dict[str, Any], extractor.extract_features(img_path))
            except Exception as e:
                print(f"  ERRORE feature {img_path}: {e}")
                continue

        feat["image"] = img_path
        feat["winner_codec"] = str(row["winner_codec"])
        feat["dataset"] = str(row["dataset"])
        rows.append(feat)

    features_df = pd.DataFrame(rows)

    # salva cache
    features_df.set_index("image")[FEATURE_COLS].to_csv(FEATURES_CACHE)

    return features_df


def train_xgboost(features_df: pd.DataFrame) -> tuple[XGBClassifier, LabelEncoder, float]:
    """Addestra XGBoost e valuta sull'hold-out set."""

    X = features_df[FEATURE_COLS].values
    y_raw = np.array(features_df["winner_codec"].values)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    print(f"\nClassi: {classes}")
    print(f"Distribuzione:\n{pd.Series(y_raw).value_counts()}")

    # split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} immagini, Test: {len(X_test)} immagini")

    # addestra XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # valuta
    y_pred = model.predict(X_test)
    accuracy = float((y_pred == y_test).mean())
    print(f"\nAccuracy hold-out: {accuracy:.3f} ({accuracy * 100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\nCross-validation 5-fold: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # feature importance
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=False)
    print("\nFeature importance:")
    for feat, imp in importance.items():
        print(f"  {feat:20s}: {imp:.4f}")

    # salva modello e label encoder
    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "features": FEATURE_COLS}, MODEL_PATH)
    print(f"\nModello salvato in {MODEL_PATH}")

    # plot feature importance + confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    importance.plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Feature Importance XGBoost")
    axes[0].set_ylabel("Importance")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)

    cm = confusion_matrix(y_test, y_pred)
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_xticklabels(classes, rotation=45)
    axes[1].set_yticklabels(classes)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[1].text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "xgboost_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot salvato in {plot_path}")

    return model, le, accuracy


if __name__ == "__main__":
    print("Caricamento dati oracolo...")
    df = pd.read_csv(ORACLE_CSV)
    df = df[df["winner_codec"] != "ERROR"].reset_index(drop=True)
    print(f"Immagini valide: {len(df)}")

    print("\nEstrazione feature spaziali...")
    features_df = extract_all_features(df)
    print(f"Feature estratte: {len(features_df)}")

    print("\nTraining XGBoost...")
    model, le, accuracy = train_xgboost(features_df)

    print(f"\n{'=' * 50}")
    print(f"RISULTATO FINALE: Accuracy = {accuracy:.1%}")
    print(f"{'=' * 50}")