#!/usr/bin/env python3
"""
CLI del selettore adattivo.
Uso: python cli.py --domain image --max-bpp 0.3 --max-lpips 0.3
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.expanduser("~/tesi"))
from src.selector.selector import AdaptiveSelector, UserConstraints


def main():
    parser = argparse.ArgumentParser(
        description="Selettore adattivo del metodo di compressione ottimo."
    )
    parser.add_argument(
        "--domain",
        choices=["image", "audio", "video"],
        required=True,
        help="Dominio del segnale da comprimere",
    )
    parser.add_argument(
        "--max-bpp",
        type=float,
        default=None,
        help="Bitrate massimo (bpp per immagini, kbps per audio)",
    )
    parser.add_argument(
        "--min-psnr", type=float, default=None, help="PSNR minimo richiesto (dB)"
    )
    parser.add_argument(
        "--max-lpips",
        type=float,
        default=None,
        help="LPIPS massimo (qualità percettiva)",
    )
    parser.add_argument(
        "--max-enc-ms",
        type=float,
        default=None,
        help="Latenza massima di codifica (ms)",
    )
    parser.add_argument(
        "--show-pareto",
        action="store_true",
        help="Mostra la frontiera di Pareto completa",
    )

    args = parser.parse_args()

    selector = AdaptiveSelector()

    constraints = UserConstraints(
        domain=args.domain,
        max_bpp=args.max_bpp,
        min_psnr=args.min_psnr,
        max_lpips=args.max_lpips,
        max_enc_ms=args.max_enc_ms,
    )

    print(selector.explain(constraints))

    # dopo selector.explain(constraints) aggiungi una nota per l'audio
    if args.domain == "audio":
        print("(Nota: PSNR = PESQ score, LPIPS = Mel Distance per il dominio audio)")

    if args.show_pareto:
        print("\n--- Frontiera di Pareto ---")
        pareto = selector.get_pareto_frontier(args.domain, max_enc_ms=args.max_enc_ms)
        cols = ["codec", "param_value", "bpp", "psnr", "lpips", "enc_ms"]
        available = [c for c in cols if c in pareto.columns]
        print(pareto[available].to_string(index=False))


if __name__ == "__main__":
    main()
