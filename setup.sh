#!/bin/bash
# ============================================================================
# Script di Setup per l'ambiente di test R-D-E
# Scarica i codec neurali ufficiali, i pesi pre-addestrati e applica le patch.
# ============================================================================

echo "=== Inizio Setup Codec Esterni ==="
mkdir -p external_codecs
cd external_codecs

# 1. TCM
echo "-> Scaricamento TCM..."
git clone https://github.com/jmliu206/LIC_TCM.git
mkdir -p LIC_TCM/checkpoints
gdown "1UbfQFsrr-Z6SrvZvpX4p1QPta5FCORZ5" -O LIC_TCM/checkpoints/tcm_mse_0013.pth.tar
gdown "1x2rfIQAv8RsjM3zEByDdOZJtEcPU5XZT" -O LIC_TCM/checkpoints/tcm_mse_0035.pth.tar

# 2. DCAE
echo "-> Scaricamento DCAE..."
git clone https://github.com/LabShuHangGU/DCAE.git
mkdir -p DCAE/checkpoints
gdown "1kXfvxsljdN3EfXDGqzknFc2Ecsgf8qgS" -O DCAE/checkpoints/dcae_mse_0013.pth.tar
gdown "1JE0SO876a-btXzOQLTilj7D0vJdePlB4" -O DCAE/checkpoints/dcae_mse_0035.pth.tar

# 3. ELIC (con patch per CompressAI >= 1.2.0)
echo "-> Scaricamento e Patching ELIC..."
git clone https://github.com/VincentChandelier/ELiC-ReImplemetation.git elic
cd elic
gdown "1uuKQJiozcBfgGMJ8CfM6lrXOZWv6RUDN" -O elic_0450.pth.tar
gdown "1s544Uxv0gBY3WvKBcGNb3Fb22zfmd9PL" -O elic_0150.pth.tar
gdown "1Moody9IR8CuAGwLCZ_ZMTfZXT0ehQhqc" -O elic_0032.pth.tar
gdown "1VNE7rx-rBFLnNFkz56Zc-cPr6xrBBJdL" -O elic_0008.pth.tar

# Patch degli import
sed -i 's/from compressai.models.priors import CompressionModel, GaussianConditional/from compressai.models.priors import CompressionModel\nfrom compressai.entropy_models import GaussianConditional/' Network.py
sed -i 's/from compressai.ops import ste_round/try:\n    from compressai.ops import ste_round\nexcept ImportError:\n    from compressai.ops import quantize_ste as ste_round/' Network.py
sed -i 's/from compressai.models.utils import conv, deconv, update_registered_buffers/try:\n    from compressai.models.utils import conv, deconv, update_registered_buffers\nexcept ImportError:\n    from compressai.layers import conv, deconv\n    from compressai.models.utils import update_registered_buffers/' Network.py
cd ..

# 4. WavTokenizer
echo "-> Scaricamento WavTokenizer..."
git clone https://github.com/jishengpeng/WavTokenizer.git
mkdir -p ../models/wavtokenizer
wget -O ../models/wavtokenizer/wavtokenizer_medium_music_audio_320_24k_v2.ckpt "https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/resolve/main/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
wget -O ../models/wavtokenizer/config.yaml "https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/resolve/main/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"

cd ..
echo "=== Setup Completato! ==="