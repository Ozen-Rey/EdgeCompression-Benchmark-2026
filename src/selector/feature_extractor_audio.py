import os
import time
import librosa
import numpy as np
from typing import Dict, Tuple

class AudioFeatureExtractor:
    """
    Estrattore di feature spettrali e temporali (11-D) per il routing XGBoost Audio.
    Ottimizzato per vincoli Edge: sottocampionamento a 16kHz e cap a 5 secondi.
    """
    def __init__(self):
        # Definiamo l'ordine esatto per XGBoost
        self.feature_names = [
            "zcr_mean",                 # Quante volte l'onda attraversa lo 0 (rumore vs voce)
            "zcr_var",
            "rms_mean",                 # Energia/Volume del segnale
            "spectral_centroid_mean",   # "Baricentro" (suoni cupi vs brillanti)
            "spectral_centroid_var",
            "spectral_bandwidth_mean",  # Larghezza di banda
            "spectral_rolloff_mean",    # Frequenza sotto la quale si concentra l'85% dell'energia
            "mfcc_1_mean",              # Inviluppo spettrale (Timbro)
            "mfcc_2_mean",
            "mfcc_3_mean",
            "mfcc_4_mean",
        ]

    def extract_features(self, audio_path: str) -> Tuple[Dict[str, float], np.ndarray]:
        if not os.path.exists(audio_path):
            raise ValueError(f"File non trovato: {audio_path}")

        # TRUCCO EDGE: Forziamo 16kHz e max 5 secondi. 
        # Caricamento ultra-rapido.
        y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=5.0)
        
        # Se l'audio è vuoto o troppo corto, gestiamo l'errore elegantemente
        if len(y) == 0:
            y = np.zeros(16000)

        features = {}

        # 1. Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_var"] = float(np.var(zcr))

        # 2. RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features["rms_mean"] = float(np.mean(rms))

        # 3. Spectral Features
        # Il centroid indica se il suono è spostato verso gli alti (difficile) o i bassi (facile)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid_mean"] = float(np.mean(cent))
        features["spectral_centroid_var"] = float(np.var(cent))

        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features["spectral_bandwidth_mean"] = float(np.mean(bw))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))

        # 4. MFCC (Mel-Frequency Cepstral Coefficients)
        # Ne estraiamo 5, scartiamo il primo (indice 0) che è solo l'energia (già misurata da RMS)
        # I coefficienti da 1 a 4 sono perfetti per catturare il timbro (es. per riconoscere la voce umana)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)[1:] 
        for i in range(4):
            features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))

        # Array per XGBoost: blindato sull'ordine della lista master
        feature_array = np.array([features[name] for name in self.feature_names], dtype=np.float32)

        return features, feature_array


if __name__ == "__main__":
    extractor = AudioFeatureExtractor()
    
    # Prendi un file a caso scaricato poco fa
    # Modifica questo path se non corrisponde a un tuo file scaricato
    test_file = os.path.expanduser("~/tesi/datasets_audio/esc50_sample/1-100032-A-0.wav")
    
    if os.path.exists(test_file):
        start = time.perf_counter()
        feats, array = extractor.extract_features(test_file)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        print(f"✅ 11 Feature Audio estratte in {elapsed_ms:.2f} ms")
        print("Feature dizionario:")
        for k, v in feats.items():
            print(f"  {k:25}: {v:10.4f}")
        print(f"\nArray XGBoost shape   : {array.shape}")
    else:
        print(f"❌ File di test non trovato: {test_file}")