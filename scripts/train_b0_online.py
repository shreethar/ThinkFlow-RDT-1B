#!/usr/bin/env python
import argparse
import os
import sys
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
    def __init__(self, dataset_dir, num_episodes=100, pred_horizon=64):
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
    
    # 2. Generate answer tokens
    tokenizer = processor.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    generated_ids = vlm.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
        eos_token_id=im_end_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # 3. Re-run full generated sequence to get hidden states
    B = generated_ids.shape[0]
    prompt_len = inputs["input_ids"].shape[1]
    new_len = generated_ids.shape[1]
    orig_mm = inputs["mm_token_type_ids"]
    pad_len = new_len - prompt_len
    if pad_len > 0:
        pad_zeros = torch.zeros(B, pad_len, dtype=orig_mm.dtype, device=orig_mm.device)
        mm_token_type_ids = torch.cat([orig_mm, pad_zeros], dim=1)
    else:
        mm_token_type_ids = orig_mm

    out = vlm(
        input_ids=generated_ids,
        attention_mask=(generated_ids != tokenizer.pad_token_id).long(),
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=mm_token_type_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    
    hidden = out.hidden_states[24] # [B, L_total, 2560]
    
    # 4. Locate answer span
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    
    answer_tokens_list = []
    answer_masks_list = []
    decoded_answers = []
    
    B = generated_ids.shape[0]
    for b in range(B):
        ids = generated_ids[b]
        
        think_end_pos = find_subsequence(ids, think_end_ids)
        im_end_positions = torch.where(ids == im_end_id)[0]
        
        if len(im_end_positions) > 0:
            end = im_end_positions[-1].item()
        else:
            end = len(ids)
            
        if think_end_pos != -1:
            start = think_end_pos
        else:
            prompt_len = inputs["attention_mask"][b].sum().item()
            start = prompt_len
            
        if start >= end:
            start = max(inputs["input_ids"].shape[1], end - 1)
            
        # Decode the answer
        decoded_ans = tokenizer.decode(ids[start:end], skip_special_tokens=True)
        decoded_answers.append(decoded_ans)
        
        x = hidden[b, start:end] # [N, 2560]
        
        # Pad or truncate to max_lang_tokens (128)
        N = x.shape[0]
        if N > max_lang_tokens:
            x = x[:max_lang_tokens]
            mask = torch.ones(max_lang_tokens, dtype=torch.bool, device=device)
        else:
            pad_len = max_lang_tokens - N
            if pad_len > 0:
                pad_tensor = torch.zeros(pad_len, 2560, dtype=x.dtype, device=device)
                x = torch.cat([x, pad_tensor], dim=0)
            mask = torch.zeros(max_lang_tokens, dtype=torch.bool, device=device)
            mask[:N] = True
            
        answer_tokens_list.append(x)
        answer_masks_list.append(mask)
        
    answer_tokens = torch.stack(answer_tokens_list, dim=0)
    answer_mask = torch.stack(answer_masks_list, dim=0)
    
    return answer_tokens, answer_mask, decoded_answers


def main():
    parser = argparse.ArgumentParser(description="B0 SFT Qwen Plan Generation and RDT Training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dataset-dir", default="dataset/droid_100/1.0.0")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--no-pretrained", action="store_true")
    
    # Test flags
    parser.add_argument("--test-a", action="store_true", help="Run Test A: Answer span generation and token shape [1, N, 2560].")
    parser.add_argument("--test-b", action="store_true", help="Run Test B: Backward pass, gradient flow, and parameter updates.")
    
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

    # 2. Load DROID Dataset
    print(f"Loading DROID dataset from {args.dataset_dir} (num_episodes={args.num_episodes})...")
    num_ep = args.num_episodes if args.num_episodes > 0 else 999999
    # For Test A & B, we only need 1 episode to extract samples
    if args.test_a or args.test_b:
        num_ep = 1
    dataset = DroidOnlineDataset(args.dataset_dir, num_episodes=num_ep, pred_horizon=cfg.model.pred_horizon)
    
    # Set batch size to 1 for Test A & B
    bs = 1 if (args.test_a or args.test_b) else args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=not (args.test_a or args.test_b),
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
        
        answer_tokens, answer_mask, decoded_answers = extract_b0_features(
            raw_batch, processor, vlm, max_lang_tokens=cfg.model.max_lang_tokens, device=device
        )
        
        print(f"Decoded Answer Plan:\n{decoded_answers[0]}")
        print(f"Extracted Hidden States Shape: {answer_tokens.shape}")
        
        assert answer_tokens.shape == (1, cfg.model.max_lang_tokens, 2560), f"Expected shape (1, {cfg.model.max_lang_tokens}, 2560), got {answer_tokens.shape}"
        print("TEST A PASSED SUCCESSFULLY!")
        return

    # ==================== TEST B ====================
    if args.test_b:
        print("\n=== Running Test B ===")
        raw_batch = next(iter(dataloader))
        
        # Save initial weights
        lang_adaptor_weight_orig = model.runner.lang_adaptor.weight.clone()
        img_adaptor_weight_orig = model.runner.img_adaptor.weight.clone()
        
        # Get trainable parameters lists
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
        if lora_params:
            lora_weight_orig = lora_params[0].clone()
        else:
            lora_weight_orig = None
            
        print("Extracting B0 features...")
        answer_tokens, answer_mask, decoded_answers = extract_b0_features(
            raw_batch, processor, vlm, max_lang_tokens=cfg.model.max_lang_tokens, device=device
        )
        
        # Prepare batch with duplicated answer tokens to both lang and img context
        batch = {
            "state": raw_batch["state"].to(device),
            "actions": raw_batch["actions"].to(device),
            "action_time_mask": raw_batch["action_time_mask"].to(device),
            "action_dim_mask": raw_batch["action_dim_mask"].to(device),
            "ctrl_freq": raw_batch["ctrl_freq"].to(device),
            "lang_tokens": answer_tokens,
            "lang_mask": answer_mask,
            "img_tokens": answer_tokens,
            "img_mask": answer_mask,
        }
        
        # Step 1: Run one optimization step so that ffn_final.fc2.weight becomes non-zero
        optimizer.zero_grad(set_to_none=True)
        metrics = model(batch)
        loss = metrics["loss"]
        print(f"Initial loss: {loss.item():.6f}")
        loss.backward()
        
        fc2_param = model.runner.model.final_layer.modules_to_save.default.ffn_final.fc2.weight
        print(f"DEBUG Step 1: fc2 grad norm = {fc2_param.grad.norm().item() if fc2_param.grad is not None else 'None'}")
        
        fc2_orig = fc2_param.clone()
        optimizer.step()
        
        fc2_change = (fc2_param - fc2_orig).abs().max().item()
        print(f"DEBUG Step 1: fc2 weight max change = {fc2_change:.6e}")
        
        # Step 2: Now run the second step to compute gradients on upstream layers
        optimizer.zero_grad(set_to_none=True)
        metrics = model(batch)
        loss = metrics["loss"]
        print(f"Loss after one step: {loss.item():.6f}")
        loss.backward()
        
        print("DEBUG Step 2: Gradients of trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm().item() if param.grad is not None else "None"
                # Print only parameters with non-zero gradient or specifically selected parameters
                if param.grad is not None and param.grad.norm().item() > 0.0 or "adaptor" in name:
                    print(f"  {name}: shape={list(param.shape)} grad_norm={grad_norm}")
        
        # Check gradients
        grad_lang = model.runner.lang_adaptor.weight.grad
        grad_img = model.runner.img_adaptor.weight.grad
        print(f"lang_adaptor grad norm: {grad_lang.norm().item() if grad_lang is not None else 'None'}")
        print(f"img_adaptor grad norm: {grad_img.norm().item() if grad_img is not None else 'None'}")
        
        assert grad_lang is not None and grad_lang.norm().item() > 0.0, "lang_adaptor gradient is zero or None!"
        assert grad_img is not None and grad_img.norm().item() > 0.0, "img_adaptor gradient is zero or None!"
        
        if lora_params:
            grad_lora = lora_params[0].grad
            print(f"lora param grad norm: {grad_lora.norm().item() if grad_lora is not None else 'None'}")
            assert grad_lora is not None and grad_lora.norm().item() > 0.0, "LoRA gradient is zero or None!"
            
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            cfg.training.max_grad_norm,
        )
        optimizer.step()
        
        # Verify weights updated
        diff_lang = (model.runner.lang_adaptor.weight - lang_adaptor_weight_orig).abs().max().item()
        diff_img = (model.runner.img_adaptor.weight - img_adaptor_weight_orig).abs().max().item()
        print(f"lang_adaptor weight max change: {diff_lang:.6e}")
        print(f"img_adaptor weight max change: {diff_img:.6e}")
        
        assert diff_lang > 0.0, "lang_adaptor weights did not change!"
        assert diff_img > 0.0, "img_adaptor weights did not change!"
        
        if lora_weight_orig is not None:
            diff_lora = (lora_params[0] - lora_weight_orig).abs().max().item()
            print(f"LoRA weight max change: {diff_lora:.6e}")
            assert diff_lora > 0.0, "LoRA weights did not change!"
            
        print("TEST B PASSED SUCCESSFULLY!")
        return

    # ==================== TEST C / FULL SFT TRAINING ====================
    print(f"\nStarting online SFT training on all {len(dataset)} windows for {args.steps} steps...")
    step = 0
    dataloader_iter = iter(dataloader)
    
    while step < args.steps:
        try:
            raw_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            raw_batch = next(dataloader_iter)
            
        # Extract Qwen features online for this batch
        answer_tokens, answer_mask, _ = extract_b0_features(
            raw_batch, processor, vlm, max_lang_tokens=cfg.model.max_lang_tokens, device=device
        )
        
        # Prepare batch with duplicated answer tokens
        batch = {
            "state": raw_batch["state"].to(device),
            "actions": raw_batch["actions"].to(device),
            "action_time_mask": raw_batch["action_time_mask"].to(device),
            "action_dim_mask": raw_batch["action_dim_mask"].to(device),
            "ctrl_freq": raw_batch["ctrl_freq"].to(device),
            "lang_tokens": answer_tokens,
            "lang_mask": answer_mask,
            "img_tokens": answer_tokens,
            "img_mask": answer_mask,
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
        
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step={step:04d} loss={value:.6f} mae={mae:.6f}")
            
        step += 1
        
    print(f"\nTraining on all {len(dataset)} windows finished successfully.")


if __name__ == "__main__":
    main()
