import os
from huggingface_hub import snapshot_download

def download_and_link():
    base_dir = "/home/ubuntu/RoboticsDiffusionTransformer/google"
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Download SigLIP
    print("Downloading SigLIP...")
    siglip_path = snapshot_download(repo_id="google/siglip-so400m-patch14-384")
    
    # 2. Download T5 XXL
    print("Downloading T5 XXL (This might take a while, ~45GB)...")
    t5_path = snapshot_download(repo_id="google/t5-v1_1-xxl")
    
    # Link
    siglip_link = os.path.join(base_dir, "siglip-so400m-patch14-384")
    if not os.path.exists(siglip_link):
        os.symlink(siglip_path, siglip_link)
        print(f"Linked SigLIP to {siglip_link}")
    else:
        print(f"SigLIP link already exists at {siglip_link}")
        
    t5_link = os.path.join(base_dir, "t5-v1_1-xxl")
    if not os.path.exists(t5_link):
        os.symlink(t5_path, t5_link)
        print(f"Linked T5 XXL to {t5_link}")
    else:
        print(f"T5 XXL link already exists at {t5_link}")

if __name__ == "__main__":
    download_and_link()
