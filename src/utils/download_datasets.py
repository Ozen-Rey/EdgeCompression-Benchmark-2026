import os
import urllib.request
from tqdm import tqdm

KODAK_BASE_URL = "http://r0k.us/graphics/kodak/kodak/"
KODAK_DIR = os.path.expanduser("~/tesi/datasets/kodak")

class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_kodak():
    os.makedirs(KODAK_DIR, exist_ok=True)
    print(f"Scaricando dataset Kodak in {KODAK_DIR}")
    
    for i in range(1, 25):
        filename = f"kodim{i:02d}.png"
        url = KODAK_BASE_URL + filename
        dest = os.path.join(KODAK_DIR, filename)
        
        if os.path.exists(dest):
            print(f"  {filename} già presente, skip")
            continue
        
        with DownloadProgress(unit="B", unit_scale=True,
                              miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
    
    print(f"\nDone. {len(os.listdir(KODAK_DIR))} immagini scaricate.")

if __name__ == "__main__":
    download_kodak()