#!/usr/bin/env python
import argparse
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face")
    parser.add_argument("--repo_id", type=str, default="shreethar/droid_100_tfds", help="HF Repo ID to upload to")
    parser.add_argument("--local_dir", type=str, default="dataset/droid_100", help="Local directory to upload")
    args = parser.parse_args()

    api = HfApi(token = "")
    
    print(f"Creating repository {args.repo_id} (if it doesn't exist)...")
    create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=True, token = "")
    
    print(f"Uploading {args.local_dir} to {args.repo_id}...")
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print("Upload complete!")

if __name__ == "__main__":
    main()
