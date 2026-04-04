import os
import pandas as pd

RAW_CSV = os.path.expanduser("~/tesi/results/audio/oracle_audio_telemetry.csv")

def simulate_policy(max_kbps, min_pesq):
    # Ricorda: audio_dist = 4.5 - PESQ. Quindi PESQ >= 3.0 significa audio_dist <= 1.5
    dist_thresh = 4.5 - min_pesq
    
    df = pd.read_csv(RAW_CSV)
    df = df[(df["status"] == "OK") & (df["kbps"] <= max_kbps)]
    
    if df.empty:
        return 0, 0
        
    opus_wins = 0
    encodec_wins = 0
    
    for file_name, group in df.groupby("file"):
        decent_opus = group[(group["audio_dist"] <= dist_thresh) & (group["codec"] == "Opus")]
        if not decent_opus.empty:
            opus_wins += 1
        else:
            # Fallback a EnCodec se Opus fallisce la soglia
            decent_encodec = group[(group["audio_dist"] <= dist_thresh) & (group["codec"] == "EnCodec")]
            if not decent_encodec.empty:
                encodec_wins += 1
            else:
                # Se fanno schifo entrambi, prende il meno peggio assoluto
                best = group.loc[group["audio_dist"].idxmin()]
                if best["codec"] == "Opus": opus_wins += 1
                else: encodec_wins += 1
                
    return opus_wins, encodec_wins

print("Simulazione Policy Eco-Mode Audio (Ricerca del Fallback Neurale)\n")
print(f"{'Max KBPS':<10} | {'Min PESQ':<10} | {'Opus Wins':<12} | {'EnCodec Wins':<12}")
print("-" * 50)

# Testiamo varie combinazioni matematiche
scenarios = [
    (48.0, 3.0),  # Quello che hai appena fatto (troppo facile per Opus)
    (24.0, 3.0),  # Dimezziamo la banda
    (24.0, 3.5),  # Pretendiamo altissima qualità
    (12.0, 3.0),  # Soffochiamo Opus a bitrate da modem 56k
    (12.0, 3.5),
]

for kbps, pesq in scenarios:
    o_wins, e_wins = simulate_policy(kbps, pesq)
    print(f"{kbps:<10} | {pesq:<10} | {o_wins:<12} | {e_wins:<12}")