import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from .metrics import compute_all_metrics_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)
transform = transforms.ToTensor()


def count_bytes(obj):
    if isinstance(obj, bytes):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(count_bytes(x) for x in obj)
    return 0


def do_pad(x, pad_multiple=64, pad_mode="right"):
    h, w = x.shape[2:]

    if pad_mode == "center":
        new_h = (h + pad_multiple - 1) // pad_multiple * pad_multiple
        new_w = (w + pad_multiple - 1) // pad_multiple * pad_multiple

        pl = (new_w - w) // 2
        pr = new_w - w - pl
        pt = (new_h - h) // 2
        pb = new_h - h - pt

        return F.pad(x, (pl, pr, pt, pb)), (pl, pr, pt, pb)

    pad_h = (pad_multiple - h % pad_multiple) % pad_multiple
    pad_w = (pad_multiple - w % pad_multiple) % pad_multiple

    return F.pad(x, (0, pad_w, 0, pad_h)), (0, pad_w, 0, pad_h)


def do_crop(x_hat, padding, h, w, pad_mode="right"):
    if pad_mode == "center":
        pl, pr, pt, pb = padding
        return F.pad(x_hat, (-pl, -pr, -pt, -pb))

    return x_hat[:, :, :h, :w]


def benchmark_neural_actual_one(
    net,
    dataset_name,
    codec_name,
    param,
    img_path,
    pad_multiple=64,
    pad_mode="right",
    params_m=None,
    disable_cudnn=False,
):
    """
    Actual bitstream benchmark:
        net.compress(...)
        net.decompress(...)

    For DCAE, disable_cudnn=True is recommended because the legacy benchmark
    already required cuDNN off during compress/decompress.
    """

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    h, w = x.shape[2:]

    x_pad, padding = do_pad(x, pad_multiple=pad_multiple, pad_mode=pad_mode)

    if device.type == "cuda":
        torch.cuda.synchronize()

    old_cudnn = torch.backends.cudnn.enabled
    if disable_cudnn:
        torch.backends.cudnn.enabled = False

    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            enc = net.compress(x_pad)
            dec = net.decompress(enc["strings"], enc["shape"])
    finally:
        torch.backends.cudnn.enabled = old_cudnn

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    time_ms = (t1 - t0) * 1000

    x_hat = do_crop(dec["x_hat"], padding, h, w, pad_mode=pad_mode).clamp(0, 1)

    total_bytes = count_bytes(enc["strings"])
    bpp = total_bytes * 8 / (h * w)

    row = {
        "dataset": dataset_name,
        "codec": codec_name,
        "param": param,
        "image": Path(img_path).name,
        "width": w,
        "height": h,
        "pixels": h * w,
        "bpp": bpp,
        "pipeline": "actual_bitstream",
        "time_ms": time_ms,
        "energy_total_j": None,
        "energy_net_j": None,
        "params_M": params_m,
    }

    row.update(compute_all_metrics_tensor(x, x_hat, img_path))
    return row


def count_params_m(net):
    return round(sum(p.numel() for p in net.parameters()) / 1e6, 3)


def prepare_net(net):
    net = net.to(device).eval()
    net.update()
    return net


def load_balle(quality):
    from compressai.zoo import bmshj2018_hyperprior

    net = bmshj2018_hyperprior(quality=quality, metric="mse", pretrained=True)
    net = prepare_net(net)
    return net, count_params_m(net)


def load_cheng(quality):
    from compressai.zoo import cheng2020_attn

    net = cheng2020_attn(quality=quality, metric="mse", pretrained=True)
    net = prepare_net(net)
    return net, count_params_m(net)


def clear_external_models_imports():
    for mod in list(sys.modules.keys()):
        if mod == "models" or mod.startswith("models."):
            del sys.modules[mod]


def load_elic(lam):
    root = Path(os.path.expanduser("~/tesi/external_codecs/elic"))
    ckpts = {
        "0.008": root / "elic_0008.pth.tar",
        "0.032": root / "elic_0032.pth.tar",
        "0.150": root / "elic_0150.pth.tar",
        "0.450": root / "elic_0450.pth.tar",
    }

    ckpt = ckpts[lam]
    if not ckpt.exists():
        raise FileNotFoundError(f"ELIC checkpoint non trovato: {ckpt}")

    clear_external_models_imports()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from Network import TestModel  # type: ignore

    net = TestModel()
    sd = torch.load(ckpt, map_location=device)
    net.load_state_dict(sd)
    net = prepare_net(net)

    return net, count_params_m(net)


def find_first_existing(paths):
    for p in paths:
        p = Path(os.path.expanduser(str(p)))
        if p.exists():
            return p
    return None


def load_tcm(lam):
    root = find_first_existing(
        [
            "~/tesi/external_codecs/LIC_TCM",
            "/tmp/LIC_TCM",
        ]
    )

    if root is None:
        raise FileNotFoundError("Cartella LIC_TCM non trovata.")

    ckpts = {
        "0.013": [
            root / "checkpoints" / "tcm_mse_0013.pth.tar",
            Path("/tmp/LIC_TCM/checkpoints/tcm_mse_0013.pth.tar"),
        ],
        "0.0035": [
            root / "checkpoints" / "tcm_mse_0035.pth.tar",
            Path("/tmp/LIC_TCM/checkpoints/tcm_mse_0035.pth.tar"),
        ],
    }

    ckpt = find_first_existing(ckpts[lam])
    if ckpt is None:
        raise FileNotFoundError(f"TCM checkpoint non trovato per lam={lam}")

    clear_external_models_imports()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models import TCM  # type: ignore

    net = TCM(
        config=[2, 2, 2, 2, 2, 2],
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0.0,
        N=64,
        M=320,
    )

    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd["state_dict"])
    net = prepare_net(net)

    return net, count_params_m(net)


def load_dcae(lam):
    root = find_first_existing(
        [
            "~/tesi/external_codecs/DCAE",
            "/tmp/DCAE",
        ]
    )

    if root is None:
        raise FileNotFoundError("Cartella DCAE non trovata.")

    ckpts = {
        "0.013": [
            root / "checkpoints" / "dcae_mse_0013.pth.tar",
            Path("/tmp/DCAE/checkpoints/dcae_mse_0013.pth.tar"),
        ],
        "0.0035": [
            root / "checkpoints" / "dcae_mse_0035.pth.tar",
            Path("/tmp/DCAE/checkpoints/dcae_mse_0035.pth.tar"),
        ],
    }

    ckpt = find_first_existing(ckpts[lam])
    if ckpt is None:
        raise FileNotFoundError(f"DCAE checkpoint non trovato per lam={lam}")

    clear_external_models_imports()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models import DCAE  # type: ignore

    net = DCAE()
    sd_raw = torch.load(ckpt, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd_raw["state_dict"].items()}
    net.load_state_dict(sd)
    net = prepare_net(net)

    return net, count_params_m(net)
