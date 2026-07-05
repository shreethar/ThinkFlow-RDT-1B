#!/usr/bin/env python
import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, get_scheduler

from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT
from thinkflow_rdt.train import create_optimizer, seed_everything


class DroidOnlineDataset(Dataset):
    def __init__(self, dataset_dir, num_episodes=100, pred_horizon=64, precomputed_path=None):
        self.pred_horizon = pred_horizon
        self.precomputed_kvs = None
        if precomputed_path and os.path.exists(precomputed_path):
            print(f"Loading precomputed Qwen KV features from {precomputed_path}...")
            self.precomputed_kvs = torch.load(precomputed_path)
        
        print("Loading tensorflow_datasets...")
        import tensorflow as tf
        # Disable TensorFlow GPU visibility to prevent it from hogging VRAM
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception as e:
            print(f"Warning: Could not disable TensorFlow GPU: {e}")
        import tensorflow_datasets as tfds
        
        # Download dataset if it doesn't exist locally
        if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
            print(f"Dataset not found at {dataset_dir}. Downloading from Hugging Face...")
            from huggingface_hub import snapshot_download
            repo_id = "shreethar/droid_100_tfds" 
            # We download to dataset/droid_100 because the repo contains the 1.0.0 folder
            parent_dir = os.path.dirname(dataset_dir)
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=parent_dir)
            print("Download complete!")
            
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
        sample = self.samples[index].copy()
        if self.precomputed_kvs is not None:
            sample["qwen_kv"] = self.precomputed_kvs[index]
        return sample


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
    
    if "qwen_kv" in samples[0]:
        batch["qwen_kv"] = torch.stack([s["qwen_kv"] for s in samples], dim=0)
        
    return batch


def find_subsequence(sequence, subsequence):
    seq_list = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
    sub_len = len(subsequence)
    for i in range(len(seq_list) - sub_len + 1):
        if seq_list[i : i + sub_len] == subsequence:
            return i + sub_len
    return -1


@torch.no_grad()
def extract_b0_features(batch, processor, vlm, max_lang_tokens=128, device="cuda"):
    instructions = batch["instructions"]
    primary_images_np = batch["primary_images"]
    wrist_images_np = batch["wrist_images"]
    
    images_list = []
    texts_list = []
    
    # 1. Format messages for each sample (prompt: no extra prompt, just instruction)
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
        
    # Preprocess inputs
    inputs = processor(
        text=texts_list,
        images=images_list,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # 2. Generate answer tokens and extract KV cache
    tokenizer = processor.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    out = vlm.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
        return_dict_in_generate=True,
        eos_token_id=im_end_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    generated_ids = out.sequences
    past_key_values = out.past_key_values
    
    if isinstance(past_key_values, tuple):
        K = past_key_values[7][0]
        V = past_key_values[7][1]
    elif hasattr(past_key_values, "key_cache"):
        K = past_key_values.key_cache[7]
        V = past_key_values.value_cache[7]
    else:
        K = past_key_values.layers[7].keys
        V = past_key_values.layers[7].values
        
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    
    qwen_kv_list = []
    decoded_answers = []
    
    B = generated_ids.shape[0]
    for b in range(B):
        ids = generated_ids[b]
        
        think_end_pos = find_subsequence(ids, think_end_ids)
        
        if think_end_pos != -1:
            target_idx = think_end_pos - 1
        else:
            im_end_positions = torch.where(ids == im_end_id)[0]
            if len(im_end_positions) > 0:
                target_idx = im_end_positions[-1].item() - 1
            else:
                target_idx = len(ids) - 1
                
        # Extract KV at target_idx
        k_vec = K[b, :, target_idx, :] # [num_kv_heads, head_dim]
        v_vec = V[b, :, target_idx, :] # [num_kv_heads, head_dim]
        
        # Flatten and concat
        k_flat = k_vec.reshape(-1)
        v_flat = v_vec.reshape(-1)
        kv_concat = torch.cat([k_flat, v_flat], dim=-1) # [num_kv_heads * head_dim * 2]
        
        qwen_kv_list.append(kv_concat.unsqueeze(0))
        
        prompt_len = inputs["attention_mask"][b].sum().item()
        decoded_ans = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
        decoded_answers.append(decoded_ans)
        
    qwen_kv = torch.stack(qwen_kv_list, dim=0).to(torch.bfloat16) # [B, 1, 1024]
    
    return qwen_kv, decoded_answers


def main():
    parser = argparse.ArgumentParser(description="B0 SFT Qwen Plan Generation and RDT Training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dataset-dir", default="dataset/droid_100/1.0.0")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--precomputed-path", type=str, default="", help="Path to precomputed Qwen KV tensor.")
    
    # Test flags
    parser.add_argument("--test-a", action="store_true", help="Run Test A: Answer span generation and token shape [1, N, 2560].")
    parser.add_argument("--test-b", action="store_true", help="Run Test B: Backward pass, gradient flow, and parameter updates.")
    parser.add_argument("--fixed-batch", action="store_true", help="Extract one fixed batch to verify overfitting (resets seed per step).")
    
    args = parser.parse_args()

    # Disable cuDNN to bypass mismatched driver issue
    torch.backends.cudnn.enabled = False

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load VLM
    if not args.precomputed_path:
        vlm_model_id = "shreethar/stage1_unsloth"
        print(f"Loading VLM from {vlm_model_id}...")
        processor = AutoProcessor.from_pretrained(vlm_model_id)
        # Ensure left-padding is used for LLM generation
        processor.tokenizer.padding_side = "left"
        
        vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_model_id,
            dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            attn_implementation="sdpa"
        )
        vlm.eval()
        vlm.requires_grad_(False)
    else:
        print("Using precomputed features. Skipping VLM initialization!")
        processor, vlm = None, None

    # 2. Load DROID Dataset
    print(f"Loading DROID dataset from {args.dataset_dir} (num_episodes={args.num_episodes})...")
    num_ep = args.num_episodes if args.num_episodes > 0 else 999999
    # For Test A, B or fixed-batch, we only need 1 episode to extract samples
    print(f"Loading DROID dataset from {args.dataset_dir} (num_episodes={num_ep})...")
    dataset = DroidOnlineDataset(args.dataset_dir, num_episodes=num_ep, pred_horizon=cfg.model.pred_horizon, precomputed_path=args.precomputed_path)
    
    # Set batch size to 1 for Test A & B
    bs = 1 if (args.test_a or args.test_b) else args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=not (args.test_a or args.test_b or args.fixed_batch),
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

    # ==================== TEST A ====================
    if args.test_a:
        print("\n=== Running Test A ===")
        raw_batch = next(iter(dataloader))
        print(f"Instruction: {raw_batch['instructions'][0]}")
        
        qwen_kv, decoded_answers = extract_b0_features(
            raw_batch, processor, vlm, max_lang_tokens=cfg.model.max_lang_tokens, device=device
        )
        
        print(f"Decoded Answer Plan:\n{decoded_answers[0]}")
        print(f"Extracted KV Cache Shape: {qwen_kv.shape}")
        
        assert qwen_kv.shape == (1, 1, cfg.model.qwen_kv_dim), f"Expected shape (1, 1, {cfg.model.qwen_kv_dim}), got {qwen_kv.shape}"
        print("TEST A PASSED SUCCESSFULLY!")
        return

    # ==================== TEST B ====================
    if args.test_b:
        print("\n=== Running Test B ===")
        raw_batch = next(iter(dataloader))
        
        # Save initial weights
        qwen_adaptor_weight_orig = model.qwen_adaptor.weight.clone()
        
        # Get trainable parameters lists
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
        if lora_params:
            lora_weight_orig = lora_params[0].clone()
        else:
            lora_weight_orig = None
                    # Extract Qwen features online
        print("Extracting B0 features...")
        qwen_kv, decoded_answers = extract_b0_features(
            raw_batch, processor, vlm, max_lang_tokens=cfg.model.max_lang_tokens, device=device
        )
        print(f"Decoded Answer Plan:\n{decoded_answers[0] if decoded_answers else 'None'}")
        
        if args.test_a:
            print(f"Extracted KV Cache Shape: {qwen_kv.shape}")
            print("TEST A PASSED SUCCESSFULLY!")
            return
            
        # Prepare batch with new qwen_kv
        batch = {
            "state": raw_batch["state"].to(device),
            "actions": raw_batch["actions"].to(device),
            "action_time_mask": raw_batch["action_time_mask"].to(device),
            "action_dim_mask": raw_batch["action_dim_mask"].to(device),
            "ctrl_freq": raw_batch["ctrl_freq"].to(device),
            "lang_tokens": torch.zeros(raw_batch["state"].shape[0], 1, 2560, device=device),
            "lang_mask": torch.ones(raw_batch["state"].shape[0], 1, dtype=torch.bool, device=device),
            "img_tokens": torch.zeros(raw_batch["state"].shape[0], 1, 2560, device=device),
            "img_mask": torch.ones(raw_batch["state"].shape[0], 1, dtype=torch.bool, device=device),
            "qwen_kv": qwen_kv,
        }
        
        # Step 1: Run one optimization step so that ffn_final.fc2.weight becomes non-zero
        optimizer.zero_grad(set_to_none=True)
        metrics = model(batch)
        loss = metrics["loss"]
        print(f"Initial loss: {loss.item():.6f}")
        loss.backward()
        
        # Set learning rate manually to a non-zero value to bypass the warmup scheduler phase during Test B
        for param_group in optimizer.param_groups:
            param_group["lr"] = 1e-3

        fc2_param = model.runner.model.final_layer.modules_to_save.default.ffn_final.fc2.weight
        print(f"DEBUG Step 1: fc2 grad norm = {fc2_param.grad.norm().item() if fc2_param.grad is not None else 'None'}")
        
        fc2_orig = fc2_param.clone()
        optimizer.step()
        
        fc2_change = (fc2_param - fc2_orig).abs().max().item()
        print(f"DEBUG Step 1: fc2 weight max change = {fc2_change:.6e}")
        
        # Step 2: Run second optimization step so that lora_B parameters become non-zero
        optimizer.zero_grad(set_to_none=True)
        metrics = model(batch)
        loss = metrics["loss"]
        print(f"Loss after one step: {loss.item():.6f}")
        loss.backward()
        optimizer.step()
        
        # Step 3: Run third step to verify gradients propagate all the way to lora_A (which depends on non-zero lora_B)
        optimizer.zero_grad(set_to_none=True)
        metrics = model(batch)
        loss = metrics["loss"]
        print(f"Loss after two steps: {loss.item():.6f}")
        loss.backward()
        
        # Check gradients
        grad_qwen = model.qwen_adaptor.weight.grad
        print(f"qwen_adaptor grad norm: {grad_qwen.norm().item() if grad_qwen is not None else 'None'}")
        
        assert grad_qwen is not None and grad_qwen.norm().item() > 0.0, "qwen_adaptor gradient is zero or None!"
        
        if lora_params:
            grad_lora = lora_params[0].grad
            print(f"lora_A param grad norm: {grad_lora.norm().item() if grad_lora is not None else 'None'}")
            assert grad_lora is not None and grad_lora.norm().item() > 0.0, "LoRA A gradient is zero or None!"
            
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            cfg.training.max_grad_norm,
        )
        optimizer.step()
        
        # Verify weights updated
        diff_qwen = (model.qwen_adaptor.weight - qwen_adaptor_weight_orig).abs().max().item()
        print(f"qwen_adaptor weight max change: {diff_qwen:.6e}")
        
        assert diff_qwen > 0.0, "qwen_adaptor weights did not change!"
        
        if lora_weight_orig is not None:
            diff_lora = (lora_params[0] - lora_weight_orig).abs().max().item()
            print(f"LoRA weight max change: {diff_lora:.6e}")
            assert diff_lora > 0.0, "LoRA weights did not change!"
            
        print("TEST B PASSED SUCCESSFULLY!")
        return

    # ==================== TEST C / FULL SFT TRAINING ====================
    if args.fixed_batch:
        print(f"Extracting one fixed batch of size {args.batch_size} from DROID...")
        raw_batch = next(iter(dataloader))
        
        print(f"\nStarting overfitting loop on this real DROID batch for {args.steps} steps...")
        initial_loss = None
        for step in range(args.steps):
            start_t = time.time()
            
            # Use precomputed features if available, otherwise extract online
            if "qwen_kv" in raw_batch:
                qwen_kv = raw_batch["qwen_kv"].to(device)
            else:
                qwen_kv, decoded_answers = extract_b0_features(
                    raw_batch, processor, vlm, max_lang_tokens=cfg.model.max_lang_tokens, device=device
                )
            
            # Prepare batch with new qwen_kv
            batch = {
                "state": raw_batch["state"].to(device),
                "actions": raw_batch["actions"].to(device),
                "action_time_mask": raw_batch["action_time_mask"].to(device),
                "action_dim_mask": raw_batch["action_dim_mask"].to(device),
                "ctrl_freq": raw_batch["ctrl_freq"].to(device),
                "lang_tokens": torch.zeros(raw_batch["state"].shape[0], 1, 2560, device=device),
                "lang_mask": torch.ones(raw_batch["state"].shape[0], 1, dtype=torch.bool, device=device),
                "img_tokens": torch.zeros(raw_batch["state"].shape[0], 1, 2560, device=device),
                "img_mask": torch.ones(raw_batch["state"].shape[0], 1, dtype=torch.bool, device=device),
                "qwen_kv": qwen_kv,
            }

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
                
            end_t = time.time()
            if step % 20 == 0 or step == args.steps - 1:
                print(f"  step={step:03d} loss={value:.6f} mae={mae:.6f} time={end_t - start_t:.2f}s")
                
        print("\nOverfitting finished.")
        print(f"  initial loss: {initial_loss:.6f}")
        print(f"  final loss:   {value:.6f}")
        print(f"  final/initial:{value / max(initial_loss, 1e-5):.4f}")
        return

    print(f"\nStarting online SFT training on all {len(dataset)} windows for {args.steps} steps...")
    step = 0
    dataloader_iter = iter(dataloader)
    
    while step < args.steps:
        start_t = time.time()
        try:
            raw_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            raw_batch = next(dataloader_iter)
            
        # Use precomputed features if available, otherwise extract online
        if "qwen_kv" in raw_batch:
            qwen_kv = raw_batch["qwen_kv"].to(device)
        else:
            qwen_kv, _ = extract_b0_features(
                raw_batch, processor, vlm, max_lang_tokens=cfg.model.max_lang_tokens, device=device
            )
        
        # Prepare batch with new qwen_kv
        batch = {
            "state": raw_batch["state"].to(device),
            "actions": raw_batch["actions"].to(device),
            "action_time_mask": raw_batch["action_time_mask"].to(device),
            "action_dim_mask": raw_batch["action_dim_mask"].to(device),
            "ctrl_freq": raw_batch["ctrl_freq"].to(device),
            "lang_tokens": torch.zeros(raw_batch["state"].shape[0], 1, 2560, device=device),
            "lang_mask": torch.ones(raw_batch["state"].shape[0], 1, dtype=torch.bool, device=device),
            "img_tokens": torch.zeros(raw_batch["state"].shape[0], 1, 2560, device=device),
            "img_mask": torch.ones(raw_batch["state"].shape[0], 1, dtype=torch.bool, device=device),
            "qwen_kv": qwen_kv,
        }
        
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
        
        end_t = time.time()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step={step:04d} loss={value:.6f} mae={mae:.6f} time={end_t - start_t:.2f}s")
            
        step += 1
        
    print(f"\nTraining on all {len(dataset)} windows finished successfully.")
    
    # Save the trained weights
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, "model.pt")
    print(f"Saving trained weights to {save_path}...")
    
    # Extract only parameters that require gradients (LoRA + Qwen Adaptor)
    trainable_state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    torch.save(trainable_state_dict, save_path)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
