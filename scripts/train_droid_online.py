#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, get_scheduler

from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT
from thinkflow_rdt.train import create_optimizer, seed_everything


class DroidOnlineDataset(Dataset):
    def __init__(self, dataset_dir, num_episodes=1, pred_horizon=64):
        self.pred_horizon = pred_horizon
        
        print("Loading tensorflow_datasets...")
        import tensorflow as tf
        # Disable TensorFlow GPU visibility to prevent it from hogging VRAM
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception as e:
            print(f"Warning: Could not disable TensorFlow GPU: {e}")
        import tensorflow_datasets as tfds
        # Build dataset from local directory
        builder = tfds.builder_from_directory(dataset_dir)
        ds = builder.as_dataset(split="train")
        
        self.samples = []
        print(f"Parsing the first {num_episodes} DROID episodes...")
        for ep_idx, episode in enumerate(ds):
            if ep_idx >= num_episodes:
                break
            
            steps = list(episode["steps"])
            L = len(steps)
            print(f"Episode {ep_idx} has {L} steps.")
            
            states = []
            actions = []
            instructions = []
            primary_imgs = []
            wrist_imgs = []
            
            for step in steps:
                cart_pos = step["observation"]["cartesian_position"].numpy()
                grip_pos = step["observation"]["gripper_position"].numpy()
                state = np.concatenate([cart_pos, grip_pos], axis=-1)
                states.append(state)
                
                actions.append(step["action"].numpy())
                instructions.append(step["language_instruction"].numpy().decode("utf-8"))
                
                primary_imgs.append(step["observation"]["exterior_image_1_left"].numpy())
                wrist_imgs.append(step["observation"]["wrist_image_left"].numpy())
                
            # Build sliding windows
            for t in range(L):
                ep_actions = actions[t : t + pred_horizon]
                valid_len = len(ep_actions)
                
                if valid_len < pred_horizon:
                    pad_len = pred_horizon - valid_len
                    pad_actions = [actions[-1]] * pad_len
                    ep_actions = ep_actions + pad_actions
                
                action_time_mask = np.zeros(pred_horizon, dtype=bool)
                action_time_mask[:valid_len] = True
                
                self.samples.append({
                    "instruction": instructions[t],
                    "primary_image": primary_imgs[t],
                    "wrist_image": wrist_imgs[t],
                    "state": states[t],
                    "actions": np.array(ep_actions, dtype=np.float32),
                    "action_time_mask": action_time_mask,
                    "ctrl_freq": 15.0,
                })
        print(f"Total parsed training windows: {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        return self.samples[index]


def droid_online_collate_fn(samples):
    batch = {
        "instructions": [s["instruction"] for s in samples],
        "primary_images": [s["primary_image"] for s in samples],
        "wrist_images": [s["wrist_image"] for s in samples],
        "state": torch.stack([torch.as_tensor(s["state"], dtype=torch.float32) for s in samples]),
        "actions": torch.stack([torch.as_tensor(s["actions"], dtype=torch.float32) for s in samples]),
        "action_time_mask": torch.stack([torch.as_tensor(s["action_time_mask"], dtype=torch.bool) for s in samples]),
        "action_dim_mask": torch.ones(len(samples), 7, dtype=torch.float32),
        "ctrl_freq": torch.tensor([s["ctrl_freq"] for s in samples], dtype=torch.float32),
    }
    return batch


@torch.no_grad()
def extract_qwen_features(batch, processor, vlm, max_lang_tokens=128, device="cuda"):
    instructions = batch["instructions"]
    primary_images_np = batch["primary_images"]
    wrist_images_np = batch["wrist_images"]
    
    images_list = []
    texts_list = []
    
    for p_img_np, w_img_np, inst in zip(primary_images_np, wrist_images_np, instructions):
        primary_img = Image.fromarray(p_img_np)
        wrist_img = Image.fromarray(w_img_np)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": primary_img},
                    {"type": "image", "image": wrist_img},
                    {"type": "text", "text": inst},
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images_list.append([primary_img, wrist_img])
        texts_list.append(text)
        
    inputs = processor(
        text=texts_list,
        images=images_list,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    output = vlm(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    
    hidden_states = output.hidden_states[24] # [B, seq_len, 2560]
    
    image_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    batch_size = len(instructions)
    lang_tokens_batch = torch.zeros(batch_size, max_lang_tokens, 2560, dtype=hidden_states.dtype, device=device)
    lang_mask_batch = torch.zeros(batch_size, max_lang_tokens, dtype=torch.bool, device=device)
    img_tokens_batch = torch.zeros(batch_size, 128, 2560, dtype=hidden_states.dtype, device=device)
    img_mask_batch = torch.ones(batch_size, 128, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        hidden = hidden_states[b] # [seq_len, 2560]
        input_ids = inputs["input_ids"][b]
        is_image_token = (input_ids == image_pad_token_id)
        
        grid_thw = inputs["image_grid_thw"] # [2*B, 3]
        N1 = grid_thw[2*b].prod().item()
        N2 = grid_thw[2*b+1].prod().item()
        M1 = N1 // 4
        M2 = N2 // 4
        
        image_indices = torch.where(is_image_token)[0]
        image_indices = image_indices[:M1 + M2]
        
        cam1_indices = image_indices[:M1]
        cam2_indices = image_indices[M1 : M1 + M2]
        
        cam1_tokens = hidden[cam1_indices]
        cam2_tokens = hidden[cam2_indices]
        
        H1, W1 = grid_thw[2*b, 1].item() // 2, grid_thw[2*b, 2].item() // 2
        cam1_tokens_reshaped = cam1_tokens.view(1, H1, W1, -1).permute(0, 3, 1, 2)
        cam1_pooled = F.adaptive_avg_pool2d(cam1_tokens_reshaped.float(), (8, 8))
        cam1_pooled = cam1_pooled.permute(0, 2, 3, 1).view(64, -1).to(hidden.dtype)
        
        H2, W2 = grid_thw[2*b+1, 1].item() // 2, grid_thw[2*b+1, 2].item() // 2
        cam2_tokens_reshaped = cam2_tokens.view(1, H2, W2, -1).permute(0, 3, 1, 2)
        cam2_pooled = F.adaptive_avg_pool2d(cam2_tokens_reshaped.float(), (8, 8))
        cam2_pooled = cam2_pooled.permute(0, 2, 3, 1).view(64, -1).to(hidden.dtype)
        
        img_tokens = torch.cat([cam1_pooled, cam2_pooled], dim=0) # [128, 2560]
        img_tokens_batch[b] = img_tokens
        
        # Language
        vision_end_indices = torch.where(input_ids == vision_end_token_id)[0]
        instruction_start_idx = vision_end_indices[-1].item() + 1
        
        im_end_indices = torch.where(input_ids == im_end_token_id)[0]
        valid_im_ends = im_end_indices[im_end_indices > instruction_start_idx]
        instruction_end_idx = valid_im_ends[0].item()
        
        lang_indices = torch.arange(instruction_start_idx, instruction_end_idx)
        lang_tokens = hidden[lang_indices]
        
        num_lang = min(len(lang_indices), max_lang_tokens)
        lang_tokens_batch[b, :num_lang] = lang_tokens[:num_lang]
        lang_mask_batch[b, :num_lang] = True
        
    return lang_tokens_batch, lang_mask_batch, img_tokens_batch, img_mask_batch


def main():
    parser = argparse.ArgumentParser(description="Online SFT training and overfitting on DROID.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dataset-dir", default="dataset/droid_100/1.0.0")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to load. Set to -1 to load all.")
    parser.add_argument("--fixed-batch", action="store_true", help="If set, overfit on a single fixed batch instead of the full dataset.")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    # Disable cuDNN to bypass mismatched driver issue
    torch.backends.cudnn.enabled = False

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load VLM
    vlm_model_id = "shreethar/stage1_unsloth"
    print(f"Loading VLM from {vlm_model_id}...")
    processor = AutoProcessor.from_pretrained(vlm_model_id)
    vlm = AutoModelForImageTextToText.from_pretrained(
        vlm_model_id,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa"
    )
    vlm.eval()
    vlm.requires_grad_(False)

    # 2. Load DROID Dataset
    print(f"Loading DROID dataset from {args.dataset_dir} (num_episodes={args.num_episodes})...")
    num_ep = args.num_episodes if args.num_episodes > 0 else 999999
    dataset = DroidOnlineDataset(args.dataset_dir, num_episodes=num_ep, pred_horizon=cfg.model.pred_horizon)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.fixed_batch,
        collate_fn=droid_online_collate_fn
    )

    # 3. Construct RDT Action Model
    print("Constructing RDT action model...")
    model = SFTConditionedRDT(cfg, load_pretrained=not args.no_pretrained)
    model.to(device)
    model.train()

    optimizer = create_optimizer(model, cfg)
    scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=args.steps,
    )

    if args.fixed_batch:
        # 4. Get one fixed batch for verification of overfitting
        print(f"Extracting one fixed batch of size {args.batch_size} from DROID...")
        raw_batch = next(iter(dataloader))

        # Move basic tensors to device
        batch = {
            "state": raw_batch["state"].to(device),
            "actions": raw_batch["actions"].to(device),
            "action_time_mask": raw_batch["action_time_mask"].to(device),
            "action_dim_mask": raw_batch["action_dim_mask"].to(device),
            "ctrl_freq": raw_batch["ctrl_freq"].to(device),
        }

        # Extract VLM features online (once, for this fixed overfitting batch)
        print("Running VLM forward pass online to extract visual and language tokens...")
        lang_tokens, lang_mask, img_tokens, img_mask = extract_qwen_features(
            raw_batch, processor, vlm, max_lang_tokens=cfg.model.max_lang_tokens, device=device
        )

        batch["lang_tokens"] = lang_tokens
        batch["lang_mask"] = lang_mask
        batch["img_tokens"] = img_tokens
        batch["img_mask"] = img_mask

        # 5. Overfitting loop
        print(f"\nStarting overfitting loop on this real DROID batch for {args.steps} steps...")
        initial_loss = None
        for step in range(args.steps):
            # Reset RNG so noise added in diffusion is deterministic each step
            torch.manual_seed(cfg.seed + 999)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed + 999)

            optimizer.zero_grad(set_to_none=True)
            metrics = model(batch)
            loss = metrics["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                cfg.training.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            value = float(loss.detach().float().cpu())
            mae = float(metrics['train_target_mae'].detach().float().cpu())
            if initial_loss is None:
                initial_loss = value
            
            if step % 20 == 0 or step == args.steps - 1:
                print(f"  step={step:03d} loss={value:.6f} mae={mae:.6f}")

        print("\nOverfitting finished.")
        print(f"  initial loss: {initial_loss:.6f}")
        print(f"  final loss:   {value:.6f}")
        print(f"  final/initial:{value / max(initial_loss, 1e-12):.4f}")

        if value >= initial_loss:
            raise RuntimeError("Overfitting failed: Loss did not decrease.")
        print("PASS: Online feature extraction and overfitting check passed successfully on real DROID data.")
    else:
        # Full training loop
        print(f"\nStarting online SFT training on all {len(dataset)} windows for {args.steps} steps...")
        step = 0
        dataloader_iter = iter(dataloader)
        
        while step < args.steps:
            try:
                raw_batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                raw_batch = next(dataloader_iter)
                
            # Move basic tensors to device
            batch = {
                "state": raw_batch["state"].to(device),
                "actions": raw_batch["actions"].to(device),
                "action_time_mask": raw_batch["action_time_mask"].to(device),
                "action_dim_mask": raw_batch["action_dim_mask"].to(device),
                "ctrl_freq": raw_batch["ctrl_freq"].to(device),
            }
            
            # Extract Qwen features online for this batch
            lang_tokens, lang_mask, img_tokens, img_mask = extract_qwen_features(
                raw_batch, processor, vlm, max_lang_tokens=cfg.model.max_lang_tokens, device=device
            )
            
            batch["lang_tokens"] = lang_tokens
            batch["lang_mask"] = lang_mask
            batch["img_tokens"] = img_tokens
            batch["img_mask"] = img_mask
            
            optimizer.zero_grad(set_to_none=True)
            metrics = model(batch)
            loss = metrics["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                cfg.training.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            value = float(loss.detach().float().cpu())
            mae = float(metrics['train_target_mae'].detach().float().cpu())
            
            if step % 10 == 0 or step == args.steps - 1:
                print(f"  step={step:04d} loss={value:.6f} mae={mae:.6f}")
                
            step += 1
            
        print(f"\nTraining on all {len(dataset)} windows finished successfully.")


if __name__ == "__main__":
    main()
