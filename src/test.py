import torch
import os
import sys


def count_parameters(model):
    """Calcola il numero totale di parametri e lo converte in milioni."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Calcolo parametri in corso su {device}...\n")

    # 1. EnCodec
    try:
        from encodec.model import EncodecModel

        encodec_model = EncodecModel.encodec_model_24khz().to(device).eval()
        enc_params = count_parameters(encodec_model)
        print(f"EnCodec:      {enc_params:>6.2f} M")
        del encodec_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"EnCodec:      Errore ({e})")

    # 2. DAC
    try:
        import dac

        dac_path = dac.utils.download(model_type="24khz")
        dac_model = dac.DAC.load(str(dac_path)).to(device).eval()
        dac_params = count_parameters(dac_model)
        print(f"DAC:          {dac_params:>6.2f} M")
        del dac_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"DAC:          Errore ({e})")

    # 3. SNAC
    try:
        from snac import SNAC

        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        snac_params = count_parameters(snac_model)
        print(f"SNAC:         {snac_params:>6.2f} M")
        del snac_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"SNAC:         Errore ({e})")

    # 4. WavTokenizer (Usando i file salvati in locale)
    try:
        # Percorso della libreria clonata
        sys.path.insert(0, os.path.expanduser("~/tesi/external_codecs/WavTokenizer"))
        from decoder.pretrained import WavTokenizer  # type: ignore

        # Percorsi esatti salvati ieri nella tua cartella models
        config_path = os.path.expanduser("~/tesi/models/wavtokenizer/config.yaml")
        model_path = os.path.expanduser(
            "~/tesi/models/wavtokenizer/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
        )

        wt_model = (
            WavTokenizer.from_pretrained0802(config_path, model_path).to(device).eval()
        )
        wt_params = count_parameters(wt_model)
        print(f"WavTokenizer: {wt_params:>6.2f} M")
        del wt_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"WavTokenizer: Errore ({e})")

    print("\nFinito.")


if __name__ == "__main__":
    main()
