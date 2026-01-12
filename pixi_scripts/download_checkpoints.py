import os
import requests
from tqdm import tqdm

CHECKPOINTS = {
    "groundingdino_swint_ogc.pth": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth": "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
}

CKPT_DIR = "ckpt"

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for filename, url in CHECKPOINTS.items():
        filepath = os.path.join(CKPT_DIR, filename)
        # Simple size check could be added, but for now existence is enough
        if os.path.exists(filepath):
            print(f"{filename} already exists in {CKPT_DIR}. Skipping download.")
        else:
            print(f"Downloading {filename}...")
            try:
                download_file(url, filepath)
                print(f"Downloaded {filename}.")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)

if __name__ == "__main__":
    main()
