import cv2
import numpy as np
import os
import time
from typing import Dict, Tuple

class SpatialFeatureExtractor:
    """
    Estrattore di feature spaziali (8-D) per il routing XGBoost.
    Ottimizzato per vincoli Edge (< 15 ms su CPU).
    """
    def __init__(self):
         self.feature_names = [
            "variance",
            "entropy",
            "sobel_mean",
            "rms_contrast",
            "laplacian_var",
            "skewness",
            "kurtosis",
            "edge_density",
            "megapixels",
            "aspect_ratio",
        ]

    def extract_features(self, image_path: str) -> Tuple[Dict[str, float], np.ndarray]:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossibile leggere immagine: {image_path}")

        img_float = img.astype(np.float32)
        features = {}

        # 1. Varianza globale
        features["variance"] = float(np.var(img_float))

        # 2. Entropia Shannon
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        hist = hist / hist.sum()
        non_zero = hist[hist > 0]
        features["entropy"] = float(-np.sum(non_zero * np.log2(non_zero)))

        # 3. Magnitudo media gradienti (Sobel)
        grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        features["sobel_mean"] = float(np.mean(magnitude))

        # 4. Contrasto RMS
        features["rms_contrast"] = float(np.std(img_float / 255.0))

        # 5. Laplacian variance (blur detection)
        lap = cv2.Laplacian(img_float, cv2.CV_32F)
        features["laplacian_var"] = float(np.var(lap))

        # --- NUOVE FEATURE ---
        
        # Statistica di ordine superiore (Momenti)
        mean_val = np.mean(img_float)
        std_val = np.std(img_float) + 1e-8  # Evita divisioni per zero
        
        # 6. Skewness
        skewness = np.mean(((img_float - mean_val) / std_val) ** 3)
        features["skewness"] = float(skewness)
        
        # 7. Kurtosi (eccesso)
        kurtosis = np.mean(((img_float - mean_val) / std_val) ** 4) - 3.0
        features["kurtosis"] = float(kurtosis)

        # 8. Edge Density (Canny)
        # Soglie 100-200 sono lo standard aureo per immagini naturali
        edges = cv2.Canny(img, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        features["edge_density"] = float(edge_density)

        # 9. Risoluzione (megapixels)
        h, w = img.shape[:2]
        features["megapixels"] = float(h * w) / 1e6

        # 10. Aspect ratio
        features["aspect_ratio"] = float(w) / float(h)

        # Array per XGBoost: blindato sull'ordine della lista master
        feature_array = np.array([features[name] for name in self.feature_names], dtype=np.float32)

        return features, feature_array


if __name__ == "__main__":
    extractor = SpatialFeatureExtractor()
    test_img = os.path.expanduser("~/tesi/datasets/kodak/kodim01.png")

    if os.path.exists(test_img):
        start = time.perf_counter()
        feats, array = extractor.extract_features(test_img)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"✅ 10 Feature estratte in {elapsed_ms:.2f} ms")
        print("Feature dizionario:")
        for k, v in feats.items():
            print(f"  {k:15}: {v:10.4f}")
        print(f"\nArray XGBoost   : {array.shape}")
    else:
        print(f"❌ File di test non trovato: {test_img}")