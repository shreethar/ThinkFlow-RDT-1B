# ThinkFlow-VLA: cached-Qwen-conditioned RDT-1B

## Part 3 KV-cache training

The current full-RDT path consumes the episode-pack cache directly:

```text
cached Qwen K/V [B, 1, 2048]
        -> trainable Linear(2048, 4096)
        -> split into native RDT K and V [B, 32, 1, 64]
        -> append inside every RDT self-attention block

current state [xyz, orientation xyz, gripper_closed]
        -> [xyz, ortho6d rotation, gripper_open] in native 128-D RDT slots
        -> frozen pretrained RDT state adaptor

normalized delta actions [B, 64, 7]
        -> trainable 7-D action adaptor
        -> trainable RDT-1B with a 7-D output head
```

Language embeddings are already cached. Part 3 stores observation JPEG slots,
so the online-SigLIP entrypoint runs the frozen RDT vision encoder but never
runs Qwen in the training loop.

The production configuration is
[`configs/part3_rdt1b.yaml`](configs/part3_rdt1b.yaml). It uses train and
validation manifests, horizon 64, LR `1e-4`, W&B, and enforces an effective
global batch of 256:

```text
micro_batch_size * gradient_accumulation_steps * world_size = 256
```

The masked denoising objective is averaged per training example, so gradient
accumulation is equivalent to a real 256-example batch even when valid future
horizons differ. Validation uses an exact 256-example global,
dataset-stratified subset for comparable BC-Z/Kuka/Fractal metrics across
checkpoints, independent of worker count.

Run the real cached-data smoke test first. It uses the real Qwen/T5/JPEG cache,
a two-block miniature RDT, frozen SigLIP, offline W&B, two optimizer steps, one
validation batch, and checkpoint round-tripping. The smoke config deliberately
uses global batch 1 for speed; the production config strictly enforces 256:

```bash
HF_HUB_OFFLINE=1 WANDB_MODE=offline \
accelerate launch --num_processes 1 scripts/train_b0_online_siglip.py \
  --config configs/part3_smoke.yaml \
  --no-pretrained
```

For full RDT-1B training, make sure the RDT checkpoint is available, authenticate
W&B, and launch:

```bash
wandb login
accelerate launch scripts/train_b0_online_siglip.py \
  --config configs/part3_rdt1b.yaml
```

The plain command uses your current Accelerate configuration (this machine
currently defaults to one process). For a multi-GPU run, create a DDP or FSDP
configuration with `accelerate config`, then pass it explicitly:

```bash
accelerate launch --config_file /path/to/accelerate.yaml \
  scripts/train_b0_online_siglip.py --config configs/part3_rdt1b.yaml
```

The runner derives accumulation from world size and checks it again after the
distributed engine is prepared. If using DeepSpeed, configure
`gradient_accumulation_steps: auto`; the currently installed DeepSpeed 0.14.2
is incompatible with this environment's PyTorch 2.10, so DDP/FSDP is the viable
local choice until DeepSpeed is upgraded.

For a 3,000-update W&B trial using the complete pretrained RDT-1B setup:

```bash
wandb login
.venv/bin/accelerate launch scripts/train_b0_online_siglip.py \
  --config configs/part3_rdt1b.yaml \
  --max-steps 3000 \
  --log-every 1 \
  --output-dir outputs/part3_rdt1b_3000 \
  --wandb-run-name part3-rdt1b-3000
```

Here, a step means one optimizer update at global batch 256, not one
microbatch. This trial therefore consumes 768,000 sampled training examples.
With `--log-every 1`, console and W&B report `train/step_time_sec` and
`train/samples_per_sec` for every optimizer update. Step timing includes data
loading, frozen SigLIP inference, forward/backward, and all accumulated
microbatches, but excludes validation and checkpoint writing.

For rank-32 LoRA instead of full RDT fine-tuning, use
`configs/part3_rdt1b_lora32.yaml`:

```bash
.venv/bin/accelerate launch scripts/train_b0_online_siglip.py \
  --config configs/part3_rdt1b_lora32.yaml \
  --max-steps 3000 \
  --log-every 1 \
  --output-dir outputs/part3_rdt1b_lora32_3000 \
  --wandb-run-name part3-rdt1b-lora32-3000
```

This freezes the base RDT transformer weights and trains rank-32 adapters in
self-attention, cross-attention, and FFN projections. The Qwen KV projector,
condition/action adaptors, and 7-D final layer remain trainable; the pretrained
RDT state adaptor remains frozen.

The cache currently lives at `cache_features/part_3` in this repository (the
external path `/home/ubuntu/cache_features/part_3` is not present on this
machine).

Important cache caveat: the existing Part 3 Bridge subset contains packs where
non-unique source episode IDs merged distinct trajectories. The supplied configs
therefore exclude Bridge and still provide 560,415 train and 68,212 validation
samples from BC-Z, Kuka, and Fractal. The precompute grouping and anchor-selection
code has been fixed for future caches; regenerate Bridge before removing
`excluded_dataset_ids: [bridge]`. Existing cache files are not modified.
Legacy `uniform` fallback anchors in the other cached datasets are remapped to
the first-step anchor at load time, so a second anchor is used only when it is
actually the first gripper change.

## Legacy LoRA path

This project trains the B0 action baseline:

```text
Frozen post-SFT Qwen features
        -> trainable language/image projectors
        -> pretrained RDT-1B core with rank-32 LoRA
        -> fully trainable 7D final action layer
```

It uses the official RDT implementation as a local dependency and keeps the
frozen Qwen model outside the training loop by caching its layer-24 features.

## 1. Install

```bash
# Use the PyTorch command appropriate for the CUDA image on your machine.
pip install torch torchvision

git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git
cd RoboticsDiffusionTransformer
pip install -r requirements.txt
cd ..

cd thinkflow_rdt1b_lora
pip install -e .
```

FlashAttention is optional for correctness but strongly recommended for speed
when the official RDT environment supports it.

## 2. Verify the complete pipeline with a tiny model

Edit `rdt_repo` in `configs/tiny_smoke.yaml`, then run:

```bash
python scripts/make_synthetic_cache.py
accelerate launch scripts/train_b0.py \
  --config configs/tiny_smoke.yaml \
  --no-pretrained
```

This proves the dataset, masking, diffusion objective, LoRA injection, optimizer,
validation sampling and checkpoint saving work before downloading RDT-1B.

## 3. Cache your real post-SFT Qwen features

Copy `scripts/cache_features_template.py` and implement only two hooks:

- `build_dataset(split)` — return your stable `FixedIndexDataset`.
- `extract_features(sample)` — reuse your validated Qwen hidden-state code.

Every cached `.pt` sample must contain:

```python
{
    "lang_tokens": FloatTensor[L_text, 2560],
    "img_tokens": FloatTensor[128, 2560],
    "state": FloatTensor[7],
    "actions": FloatTensor[T <= 64, 7],
    "ctrl_freq": float,
    # optional:
    "lang_mask": BoolTensor[L_text],
    "img_mask": BoolTensor[128],
    "action_time_mask": BoolTensor[T],
    "action_dim_mask": FloatTensor[7],
}
```

Recommended feature extraction:

1. Give Qwen the current agent-view image, current wrist-view image and task
   instruction.
2. Do not include the ground-truth action text in the prompt.
3. Run a forward pass with `output_hidden_states=True` and select layer 24.
4. Extract instruction-token hidden states for `lang_tokens`.
5. Extract each camera's image tokens and reshape with `image_grid_thw`.
6. Adaptive-average-pool each camera to 8 x 8 = 64 tokens.
7. Concatenate two cameras to obtain exactly 128 image tokens.
8. Save features before the trainable 2560 -> 2048 projector.

Split by episode before caching. Do not split neighbouring frames randomly.

## 4. Configure the real run

Edit these paths in `configs/b0_rdt1b_lora.yaml`:

```yaml
rdt_repo: /path/to/RoboticsDiffusionTransformer
data:
  train_manifest: /path/to/cache/train/manifest.jsonl
  val_manifest: /path/to/cache/val/manifest.jsonl
```

The default final configuration is:

- RDT-1B: 28 blocks, hidden size 2048, 32 heads.
- Action horizon: 64.
- Diffusion training steps: 1000.
- DPM-Solver inference steps: 5.
- LoRA: rank 32, alpha 64, dropout 0.05.
- LoRA targets: self-attention qkv/proj, cross-attention q/kv/proj, FFN fc1/fc2.
- Fully trainable: Qwen condition adaptors, state adaptor and final action layer.
- Frozen: pretrained RDT base weights and all deterministic position embeddings.

Inspect the exact modules before training:

```bash
python scripts/inspect_lora_targets.py \
  --config configs/b0_rdt1b_lora.yaml
```

For 28 blocks and all seven target linears, the script should print 196 targets.

## 5. Train

Single GPU:

```bash
accelerate config
accelerate launch scripts/train_b0.py \
  --config configs/b0_rdt1b_lora.yaml
```

For an RTX 5090, start with micro-batch 1 and gradient accumulation 32. Increase
accumulation if the loss oscillates. The effective global batch size is:

```text
micro_batch * accumulation * number_of_processes
```

The code saves only the LoRA adapter, fully trainable final RDT layer and the
three external adaptors. It does not duplicate the frozen 1.2B base weights in
every checkpoint.

## 6. What is copied from the pretrained checkpoint

The loader creates a 7D RDT runner and transfers only compatible pretrained
parts:

- all 28 Transformer blocks;
- timestep and control-frequency embedders;
- state/action sequence positional embedding;
- final normalization;
- optionally the first final-layer MLP projection.

The following remain newly initialized because their dimensions or semantics
changed:

- Qwen language projector;
- Qwen visual projector;
- state/action input adaptor;
- Qwen condition position embeddings;
- final 2048 -> 7 action output projection.

This avoids pretending that T5/SigLIP or 128D unified-action interface weights
are compatible with Qwen 2560D features and LIBERO 7D actions.

## 7. Important limitations

The cache template is the only project-specific integration point because your
current Qwen3.5/Unsloth checkpoint loader and canonical dataset classes are not
part of this package. The training code itself is complete.

The direct 7D configuration assumes proprioception is also represented as seven
values. When your LIBERO state has a different width, pad or project it to seven
values in the cache adapter; do not silently truncate it.

For closed-loop LIBERO evaluation, predict 64 actions but initially execute only
4-8 before observing again. Keep the execution horizon identical in every B0-B5
ablation.

## 8. Sample actions from a saved checkpoint

```bash
python scripts/sample_cached.py \
  --config configs/b0_rdt1b_lora.yaml \
  --artifact outputs/b0_sft_rdt1b_lora/final \
  --index 0 \
  --output sampled_actions.pt
```

This reconstructs the frozen official RDT-1B backbone, loads your LoRA/interface
artifact and samples a 64 x 7 action chunk with the configured five DPM-Solver
steps.

## Model-only smoke test before building a dataset

This validates the complete RDT-1B/LoRA interface without a dataset. It creates one fixed batch of tensors with the same shapes as frozen Qwen features, performs forward/backward passes, verifies that LoRA and interface weights update, verifies that the frozen RDT backbone does not update, and checks that the fixed-batch loss decreases.

```bash
python scripts/smoke_test_model_only.py \
  --config configs/b0_rdt1b_lora.yaml \
  --steps 20 \
  --batch-size 1
```

To use one real set of frozen Qwen features instead of random features, save a `.pt` file containing:

```python
{
    "lang_tokens": Tensor[L, 2560],
    "img_tokens": Tensor[128, 2560],
    "state": Tensor[7],                 # optional; defaults to zeros
    "actions": Tensor[64, 7],           # optional; defaults to a smooth synthetic target
    "lang_mask": BoolTensor[L],         # optional
    "img_mask": BoolTensor[128],        # optional
    "action_time_mask": BoolTensor[64], # optional
    "action_dim_mask": Tensor[7],       # optional
    "ctrl_freq": 10.0,                  # optional
}
```

Then run:

```bash
python scripts/smoke_test_model_only.py \
  --config configs/b0_rdt1b_lora.yaml \
  --feature-file one_real_qwen_probe.pt \
  --steps 20
```

The script deliberately resets the diffusion RNG at every optimization step. This fixes the sampled diffusion noise, timestep, and LoRA dropout so that a decrease in loss is an unambiguous test of learnability. Disable this only after the model passes:

```bash
--no-fixed-diffusion-rng
```
