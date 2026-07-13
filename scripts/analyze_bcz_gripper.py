import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

def main():
    data_dir = Path("dataset/mock_dataset/bc_z_dataset/data").resolve()
    print(f"Loading TFDS dataset from {data_dir}")
    builder = tfds.builder_from_directory(str(data_dir))
    dataset = builder.as_data_source(split='train')
    
    # Iterate through a few episodes
    episode_count = 0
    iterator = iter(dataset)
    for _ in range(5):
        try:
            episode = next(iterator)
        except StopIteration:
            break
            
        print(f"\n--- Episode {episode_count} ---")
        steps = list(episode['steps'])
        
        sensed_close_vals = []
        target_close_0_vals = []
        target_close_all_vals = []
        
        for step in steps:
            sensed_close = step['observation']['present/sensed_close'][0]
            target_close = step['action']['future/target_close']
            
            sensed_close_vals.append(sensed_close)
            target_close_0_vals.append(target_close[0])
            target_close_all_vals.append(target_close)
            
        # Print out the sequences
        print("Step | Sensed Close (float) | Target Close [0] (int) | Target Close (all 10)")
        for i in range(len(steps)):
            sensed = sensed_close_vals[i]
            target_0 = target_close_0_vals[i]
            target_all = target_close_all_vals[i]
            print(f"{i:4d} | {sensed:19.3f} | {target_0:22d} | {target_all}")
            
        episode_count += 1

if __name__ == "__main__":
    # Disable GPU for script
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    main()
