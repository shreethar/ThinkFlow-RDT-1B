import json
from pathlib import Path

targets = {
    "bridge": ("dataset/mock_dataset/bridge_dataset/data/dataset_info.json", 25000),
    "fractal": ("dataset/mock_dataset/fractal_dataset/data/dataset_info.json", 25000),
    "kuka": ("dataset/mock_dataset/kuka_dataset/data/dataset_info.json", 25000),
    "droid": ("dataset/mock_dataset/droid_dataset/data/dataset_info.json", 15000),
    "bc_z": ("dataset/mock_dataset/bc_z_dataset/data/dataset_info.json", 15000)
}

for name, (path, target) in targets.items():
    with open(path, "r") as f:
        data = json.load(f)
    
    # usually split 0 is train
    train_split = next(s for s in data["splits"] if s["name"] == "train")
    shard_lengths = [int(x) for x in train_split["shardLengths"]]
    
    total_episodes = 0
    shards_needed = 0
    for length in shard_lengths:
        total_episodes += length
        shards_needed += 1
        if total_episodes >= target:
            break
            
    print(f"{name}: Need {shards_needed} shards to reach {total_episodes} episodes (Target: {target}). Total shards available: {len(shard_lengths)}")
