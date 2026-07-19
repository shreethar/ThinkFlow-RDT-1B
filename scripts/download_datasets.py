import json
import subprocess
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent

# Map of dataset name to (info_file, episode_target, gs_prefix)
TARGETS = {
    # "bridge": (
    #     REPO_ROOT / "dataset/mock_dataset/bridge_dataset/data/dataset_info.json",
    #     25000,
    #     "gs://gresearch/robotics/bridge_data_v2/0.0.1/bridge_data_v2-train.tfrecord-{shard:05d}-of-01024"
    # ),
    # "fractal": (
    #     REPO_ROOT / "dataset/mock_dataset/fractal_dataset/data/dataset_info.json",
    #     25000,
    #     "gs://gresearch/robotics/fractal20220817_data/0.1.0/fractal20220817_data-train.tfrecord-{shard:05d}-of-01024"
    # ),
    # "kuka": (
    #     REPO_ROOT / "dataset/mock_dataset/kuka_dataset/data/dataset_info.json",
    #     25000,
    #     "gs://gresearch/robotics/kuka/0.1.0/kuka-train.tfrecord-{shard:05d}-of-01024"
    # ),
    "droid": (
        REPO_ROOT / "dataset/mock_dataset/droid_dataset/data/dataset_info.json",
        15000,
        "gs://gresearch/robotics/droid/1.0.1/droid_101-train.tfrecord-{shard:05d}-of-02048"
    ),
    "bc_z": (
        REPO_ROOT / "dataset/mock_dataset/bc_z_dataset/data/dataset_info.json",
        15000,
        "gs://gresearch/robotics/bc_z/1.0.1/bc_z-train.array_record-{shard:05d}-of-01024"
    )
}

def main():
    for name, (info_path, target_episodes, gs_prefix) in TARGETS.items():
        print(f"--- Processing {name} ---")
        if not info_path.exists():
            print(f"Error: {info_path} does not exist. Skipping.")
            continue
            
        with open(info_path, "r") as f:
            data = json.load(f)
            
        # Extract train split
        train_split = next((s for s in data["splits"] if s["name"] == "train"), None)
        if not train_split:
            print(f"Error: Could not find train split for {name}")
            continue
            
        shard_lengths = [int(x) for x in train_split["shardLengths"]]
        
        total_episodes = 0
        shards_needed = 0
        for length in shard_lengths:
            total_episodes += length
            shards_needed += 1
            if total_episodes >= target_episodes:
                break
                
        print(f"{name}: Selected {shards_needed} shards to reach {total_episodes} episodes (Target: {target_episodes})")
        
        # Generate URIs
        uris = []
        for i in range(shards_needed):
            uri = gs_prefix.format(shard=i)
            uris.append(uri)
            
        dest_dir = info_path.parent
        print(f"Downloading {len(uris)} files to {dest_dir} ...")
        
        # Batch URIs into chunks of 100 to be safe
        chunk_size = 100
        for i in range(0, len(uris), chunk_size):
            chunk = uris[i:i + chunk_size]
            cmd = ["gsutil", "-m", "cp", "-n"] + chunk + [str(dest_dir)]
            
            print(f"Executing chunk {i//chunk_size + 1}/{(len(uris)-1)//chunk_size + 1}")
            process = subprocess.run(cmd)
            
            if process.returncode != 0:
                print(f"Error: gsutil failed with return code {process.returncode}")
                sys.exit(process.returncode)
                
        print(f"Finished {name}\n")

if __name__ == "__main__":
    main()
