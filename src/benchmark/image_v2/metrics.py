import os
import tempfile
import numpy as np
import torch
from PIL import Image
import piq
import lpips as lpips_lib
from pytorch_msssim import ms_ssim as compute_ms_ssim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)

loss_fn_lpips = lpips_lib.LPIPS(net="alex").to(device).eval()
loss_fn_dists = piq.DISTS().to(device).eval()

try:
    import ssimulacra2 as ssimulacra2_mod

    HAS_SSIMULACRA2 = True
except ImportError:
    ssimulacra2_mod = None
    HAS_SSIMULACRA2 = False


def scalar_item(v):
    """
    Alcune funzioni PIQ hanno typing poco preciso per Pylance.
    Runtime: di solito ritornano Tensor scalare.
    Fallback: se ritorna lista/tupla, prende il primo elemento.
    """
    if isinstance(v, (list, tuple)):
        v = v[0]
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


def np_to_tensor(img_np):
    if not img_np.flags.writeable:
        img_np = img_np.copy()

    return (
        torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        / 255.0
    )


def compute_all_metrics_np(orig_np, rec_np, orig_path=None):
    x = np_to_tensor(orig_np)
    x_hat = np_to_tensor(rec_np).clamp(0, 1)
    return compute_all_metrics_tensor(x, x_hat, orig_path)


def compute_all_metrics_tensor(x, x_hat, orig_path=None):
    m = {}

    with torch.no_grad():
        m["psnr"] = scalar_item(piq.psnr(x, x_hat, data_range=1.0))
        m["ssim"] = scalar_item(piq.ssim(x, x_hat, data_range=1.0))
        m["ms_ssim"] = scalar_item(compute_ms_ssim(x, x_hat, data_range=1.0))

        m["lpips"] = scalar_item(loss_fn_lpips(x * 2 - 1, x_hat * 2 - 1))
        m["dists"] = scalar_item(loss_fn_dists(x, x_hat))

        m["fsim"] = scalar_item(piq.fsim(x, x_hat, data_range=1.0))
        m["gmsd"] = scalar_item(piq.gmsd(x, x_hat, data_range=1.0))
        m["vif"] = scalar_item(piq.vif_p(x, x_hat, data_range=1.0))
        m["haarpsi"] = scalar_item(piq.haarpsi(x, x_hat, data_range=1.0))
        m["dss"] = scalar_item(piq.dss(x, x_hat, data_range=1.0))
        m["mdsi"] = scalar_item(piq.mdsi(x, x_hat, data_range=1.0))

    if HAS_SSIMULACRA2 and ssimulacra2_mod is not None and orig_path is not None:
        tmp_path = None
        try:
            rec_np = (
                (x_hat[0].detach().cpu().permute(1, 2, 0).numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            Image.fromarray(rec_np).save(tmp_path)

            m["ssimulacra2"] = ssimulacra2_mod.compute_ssimulacra2(
                str(orig_path), tmp_path
            )

        except Exception:
            m["ssimulacra2"] = None

        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        m["ssimulacra2"] = None

    return m
