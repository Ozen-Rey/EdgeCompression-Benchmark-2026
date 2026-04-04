import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser("~/tesi"))
from src.selector.feature_extractor_audio import AudioFeatureExtractor

# ================== CONFIGURAZIONE ==================
RAW_CSV = os.path.expanduser("~/tesi/results/audio/oracle_audio_telemetry.csv")
ECO_CSV = os.path.expanduser("~/tesi/results/audio/oracle_audio_eco_winners.csv")
FEATURES_CACHE = os.path.expanduser("~/tesi/results/audio/features_audio_cache.csv")
MODEL_PATH = os.path.expanduser("~/tesi/results/audio/xgboost_audio_model.pkl")

# Percorsi dove cercare i file audio originali per l'estrazione delle feature
DATASETS_DIRS = [
    os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    os.path.expanduser("~/tesi/datasets_audio/esc50_sample")
]

# --- POLICY ECO-MODE AUDIO ---
MAX_KBPS = 12.0
AUDIO_DIST_THRESH = 1.5  # PESQ >= 3.0

FEATURE_COLS = [
    "zcr_mean", "zcr_var", "rms_mean", "spectral_centroid_mean", 
    "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_rolloff_mean", 
    "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean"
]

# ================== 1. APPLICAZIONE POLICY ==================
def apply_eco_policy():
    print("Applicazione Policy Eco-Mode Audio...")
    df = pd.read_csv(RAW_CSV)
    
    # Filtro 1: Rimuovi errori e tieni solo configurazioni ammissibili per banda
    df = df[(df["status"] == "OK") & (df["kbps"] <= MAX_KBPS)]
    
    if df.empty:
        raise ValueError("Nessun dato valido trovato nel CSV sotto i 48 kbps.")
        
    winners = []
    for file_name, group in df.groupby("file"):
        # Cerca un codec Classico (Opus) che sia "decente" (audio_dist <= 1.5)
        decent_classical = group[(group["audio_dist"] <= AUDIO_DIST_THRESH) & (group["codec"] == "Opus")]
        
        if not decent_classical.empty:
            # Trovato! Vinci il risparmio energetico. Scegli quello con distorsione minore.
            best_match = decent_classical.loc[decent_classical["audio_dist"].idxmin()]
        else:
            # Fallback Neurale: Opus fa schifo su questo file, proviamo EnCodec
            decent_neural = group[(group["audio_dist"] <= AUDIO_DIST_THRESH) & (group["codec"] == "EnCodec")]
            if not decent_neural.empty:
                best_match = decent_neural.loc[decent_neural["audio_dist"].idxmin()]
            else:
                # Caso limite: nulla è decente, salva il salvabile prendendo il minimo assoluto
                best_match = group.loc[group["audio_dist"].idxmin()]
                
        winners.append(best_match)
        
    winners_df = pd.DataFrame(winners)
    os.makedirs(os.path.dirname(ECO_CSV), exist_ok=True)
    winners_df.to_csv(ECO_CSV, index=False)
    
    print(f"Totale immagini/audio processati: {len(winners_df)}")
    print("\nDistribuzione Vincitori (Eco-Mode):")
    print(winners_df["codec"].value_counts())
    return winners_df

# ================== 2. ESTRAZIONE FEATURE ==================
def get_file_path(filename):
    """Cerca il file audio originale nelle cartelle dei dataset."""
    for d in DATASETS_DIRS:
        for ext in ['.wav', '.flac']:
            # Cerca il file in modo flessibile
            paths = list(Path(d).rglob(f"{Path(filename).stem}{ext}"))
            if paths: 
                return str(paths[0])
    return None

def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(FEATURES_CACHE):
        print(f"\nCache feature audio trovata ({FEATURES_CACHE}). Caricamento...")
        return pd.read_csv(FEATURES_CACHE)
    
    print("\nEstrazione feature audio in corso (Numba warm-up iniziale)...")
    extractor = AudioFeatureExtractor()
    rows = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_path = get_file_path(row["file"])
        if file_path is None:
            print(f"ATTENZIONE: File sorgente non trovato per {row['file']}")
            continue
            
        try:
            feats, _ = extractor.extract_features(file_path)
            feats["file"] = row["file"]
            feats["codec"] = row["codec"]
            rows.append(feats)
        except Exception as e:
            print(f"Errore nell'estrazione di {row['file']}: {e}")
            
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(FEATURES_CACHE, index=False)
    return feat_df

# ================== 3. TRAINING XGBOOST ==================
def train_xgboost(features_df: pd.DataFrame):
    print("\nTraining XGBoost Audio...")
    X = features_df[FEATURE_COLS].values
    
    le = LabelEncoder()
    y = le.fit_transform(features_df["codec"])
    classes = le.classes_
    
    # Parametri già robusti e testati (max_depth 3 per evitare overfitting su dataset piccoli)
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="mlogloss", verbosity=0
    )
    
    # Hold-out validation (80-20)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    print(f"\nClassi: {classes}")
    print("\nClassification Report (Hold-out 20%):")
    print(classification_report(y_te, y_pred, target_names=classes))
    
    # Cross Validation a 5 fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"==================================================")
    print(f"RISULTATO CV 5-FOLD: {scores.mean():.3f} ± {scores.std():.3f} ({scores.mean()*100:.1f}%)")
    print(f"==================================================")
    
    # Importanza delle Feature
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop Feature Importance:")
    for feat, imp in importance.head(5).items():
        print(f"  {feat:25}: {imp:.4f}")
        
    joblib.dump({"model": model, "label_encoder": le, "features": FEATURE_COLS}, MODEL_PATH)
    print(f"\nModello salvato in {MODEL_PATH}")

if __name__ == "__main__":
    if not os.path.exists(RAW_CSV):
        print(f"ERRORE: CSV Telemetria non trovato in {RAW_CSV}.")
        print("Devi aspettare che oracle_audio.py finisca!")
        sys.exit(1)
        
    winners_df = apply_eco_policy()
    
    # Resettiamo l'indice prima di fonderlo con le feature per evitare problemi (come con le immagini)
    winners_df = winners_df.reset_index(drop=True)
    
    features_df = extract_all_features(winners_df)
    train_xgboost(features_df)