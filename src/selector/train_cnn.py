"""
Training della Micro-CNN per il selettore adattivo.
Ablation study: confronto con XGBoost su stesso hold-out set.
Input: oracle_v2_multi.csv + patch a risoluzione nativa (128x128)
Output: modello salvato + report accuracy + plot training
"""

import os
import sys
from typing import Optional
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.expanduser("~/tesi"))

ORACLE_CSV = os.path.expanduser("~/tesi/results/oracle/oracle_raw_telemetry_eco_winners_3class.csv")
RESULTS_DIR = os.path.expanduser("~/tesi/results/selector")
MODEL_PATH = os.path.expanduser("~/tesi/results/selector/cnn_model.pth")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────────


class CodecDataset(Dataset):
    """
    Dataset di patch a risoluzione nativa con data augmentation.
    Preserva le alte frequenze spaziali vitali per la compressione.
    """

    def __init__(self, img_paths: list[str], labels: list[int], is_train: bool = True) -> None:
        self.img_paths = img_paths
        self.labels = labels
        self.is_train = is_train

        # Pipeline per il Training (Augmentation estrema e random crop)
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(IMG_SIZE, pad_if_needed=True, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        # Pipeline per il Test (Crop centrale deterministico)
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        
        # Usiamo cast per placare l'ansia di Pylance
        x = cast(torch.Tensor, self.transform(img))
        
        return x, self.labels[idx]


# ── Architettura ─────────────────────────────────────────────────


class MicroCNN(nn.Module):
    """
    Micro-CNN per routing del codec.
    Input: patch 3x128x128
    Output: logit per N classi codec
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 16 -> 8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Training ─────────────────────────────────────────────────────


def train_epoch(
    model: MicroCNN,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model: MicroCNN, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total


def train_cnn(
    df: pd.DataFrame,
) -> tuple[MicroCNN, LabelEncoder, float]:
    """Addestra la Micro-CNN e valuta sull'hold-out set."""

    img_paths = df["image"].tolist()
    y_raw = np.array(df["codec"].values)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    n_classes = len(classes)

    print(f"\nClassi: {classes}")
    print(f"Distribuzione:\n{pd.Series(y_raw).value_counts()}")

    # split 80/20 — stesso random_state di XGBoost per confronto equo
    img_paths_np = np.array(img_paths)
    y_np = np.array(y)

    idx = np.arange(len(img_paths))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y_np
    )

    train_paths = img_paths_np[idx_train].tolist()
    test_paths = img_paths_np[idx_test].tolist()
    y_train = y_np[idx_train].tolist()
    y_test_list = y_np[idx_test].tolist()

    print(f"\nTrain: {len(train_paths)} immagini, Test: {len(test_paths)} immagini")

    # Iniettiamo il flag is_train per discriminare i due loader
    train_loader = DataLoader(
        CodecDataset(train_paths, y_train, is_train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        CodecDataset(test_paths, y_test_list, is_train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = MicroCNN(n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining su {DEVICE} per {EPOCHS} epoche...")
    train_losses = []
    val_accs = []
    best_acc = 0.0
    best_state: Optional[dict] = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        acc = eval_epoch(model, test_loader)
        scheduler.step()
        train_losses.append(loss)
        val_accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={loss:.4f} | val_acc={acc:.3f}")

    # carica il miglior modello
    if best_state is not None:
        model.load_state_dict(best_state)

    # valutazione finale
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y_batch in test_loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_true)))
    print(f"\nAccuracy hold-out (best): {accuracy:.3f} ({accuracy * 100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=classes))

    # salva modello
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "label_encoder": le, "n_classes": n_classes},
        MODEL_PATH,
    )
    print(f"\nModello salvato in {MODEL_PATH}")

    # plot training curve + confusion matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_losses, color="steelblue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_accs, color="darkorange")
    axes[1].axhline(best_acc, color="red", linestyle="--", label=f"Best: {best_acc:.3f}")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_true, all_preds)
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks(range(len(classes)))
    axes[2].set_yticks(range(len(classes)))
    axes[2].set_xticklabels(classes, rotation=45)
    axes[2].set_yticklabels(classes)
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    axes[2].set_title("Confusion Matrix CNN")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[2].text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "cnn_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot salvato in {plot_path}")

    return model, le, accuracy


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Caricamento dati oracolo...")
    df = pd.read_csv(ORACLE_CSV)
    df = df[df["codec"] != "ERROR"].reset_index(drop=True)
    print(f"Immagini valide: {len(df)}")

    print(f"\nDevice: {DEVICE}")
    print(f"Architettura: MicroCNN 128x128 → {EPOCHS} epoche")

    model, le, accuracy = train_cnn(df)

    print(f"\n{'=' * 50}")
    print(f"RISULTATO FINALE CNN: Accuracy = {accuracy:.1%}")
    print(f"{'=' * 50}")