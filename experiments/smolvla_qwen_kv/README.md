# Qwen-KV-conditioned SmolVLA

This is a separate experiment. It does not patch the installed LeRobot package
and does not change any RDT source, configuration, cache, notebook, or script.

## Architecture

The cached feature is split exactly as:

```text
qwen_kv [B, 1, 2048]
  ├─ first 1024  -> Qwen K
  └─ second 1024 -> Qwen V
```

SmolVLA 0.6 does not pass a generic `[B,L,expert_hidden]` tensor into its
Action Expert cross-attention. Its VLM cache is represented per KV head as
`[B,L,5,64]`, or 320 values per token. Therefore, every Action Expert
cross-attention layer receives its own pair of learned adapters:

```text
Qwen K [B,1,1024] -> Linear_layer -> [B,1,5,64]
Qwen V [B,1,1024] -> Linear_layer -> [B,1,5,64]
```

Those tensors are appended to that layer's normal image/language/state K/V,
then passed through the pretrained expert K/V projections. The Qwen token has a
learnable attention-logit bias per layer and head, initialized to `-4` so the
pretrained policy is disturbed gradually.

Appending is important. Replacing all normal conditioning with one Qwen token
would produce a one-element attention softmax, whose value is always 1. The
query and key would then have no effect and the key adapter could not learn.

The model uses LeRobot's native LIBERO representation:

- State: 8D `[x,y,z, axis_angle_x,axis_angle_y,axis_angle_z, finger0,finger1]`.
- Target: 7D `[dx,dy,dz, dRx,dRy,dRz, raw_gripper]`.
- Images: current agent view (`observation.images.image`) and wrist view
  (`observation.images.image2`), decoded losslessly.
- Language: the original LIBERO instruction through SmolVLM's tokenizer.
- Extra condition: cached Qwen `[K|V]` under `batch["qwen_kv"]`.

The existing feature shards physically store the earlier 11D/10D orthogonal-6D
representation. The new loader decodes it back to 8D/7D when each shard is
loaded. This is the inverse of the cache conversion: absolute ortho6D becomes
an axis-angle state and relative ortho6D becomes the original normalized
LIBERO rotation command. Translation, both finger positions, and raw gripper
command are copied unchanged.

The serialized `smolvla_base` config contains generic 6D state/action feature
placeholders because the base model is robot-agnostic. For LIBERO fine-tuning,
the active schema is replaced by 8D/7D. SmolVLA pads both vectors to its
internal 32D representation, so this does not resize the pretrained projection
weights.

## Full four-suite fine-tuning run

From the repository root:

```bash
bash experiments/smolvla_qwen_kv/run_all_suites.sh
```

Or invoke the trainer directly:

```bash
.venv/bin/python -m experiments.smolvla_qwen_kv.train_cached \
  --pretrained lerobot/smolvla_base \
  --cache-root cache_features_libero_b0_raw_ortho6d \
  --suites libero_10 libero_spatial libero_goal libero_object \
  --output-dir outputs/smolvla_base_qwen_kv_all_suites \
  --batch-size 256 \
  --gradient-accumulation 1 \
  --steps 100000 \
  --bf16
```

The first run computes combined state/action mean and standard deviation across
all valid cached targets in all four suites and saves them as `cache_stats.pt`
in the output directory. Later runs reuse that file. Shards are shuffled as one
combined stream, so suites are sampled in proportion to their cached sample
counts. Cached images are not normalized by these statistics; SmolVLA applies
its own image preparation.

By default, the VLM and vision encoder are frozen. The Action Expert, flow
matching projections, state projection, per-layer Qwen K/V adapters, and
attention biases are trainable. Pass `--train-vlm` only if a later controlled
experiment shows that expert-only adaptation is insufficient.

Checkpoints contain:

- `model.safetensors` with both pretrained and new adapter parameters;
- the custom `config.json` (`type: smolvla_qwen_kv`);
- matching preprocessor/postprocessor statistics;
- optimizer and scheduler state in `training_state.pt`.

Load a saved policy only after importing this package, so the custom config
type is registered:

```python
from experiments.smolvla_qwen_kv import KVSmolVLAPolicy

policy = KVSmolVLAPolicy.from_pretrained(
    "outputs/smolvla_base_qwen_kv_all_suites/checkpoint-001000"
)
```

Every training or inference batch must include `qwen_kv` with shape
`[B,1,2048]`. During online LIBERO rollout this feature must be recomputed from
the current observation/instruction using the same Qwen extraction procedure
that created the cache.

## LeRobot policy and cached-dataset integration

The custom policy is installed as the editable third-party distribution
`lerobot_policy_smolvla_qwen_kv`. LeRobot's normal plugin discovery therefore
registers the `smolvla_qwen_kv` policy type, resolves `KVSmolVLAPolicy` through
its normal policy factory, and uses the official SmolVLA processors.

It is already installed in this workspace's `.venv`. To reinstall it in a new
environment without downloading build dependencies:

```bash
UV_CACHE_DIR=/tmp/smolvla-uv-cache uv pip install \
  --python .venv/bin/python --no-deps --no-build-isolation -e \
  experiments/lerobot_policy_smolvla_qwen_kv
```

The cache is exposed as a LeRobot-compatible `IterableDataset` without copying
the samples to a new dataset. Use `cached_libero_qwen:all`, or select suites
with a comma-separated ID such as
`cached_libero_qwen:libero_goal,libero_object`. Always set
`--dataset.streaming=true`.

Checkpoint-014000 closed-loop smoke test (task 0, two episodes in each suite,
with videos):

```bash
bash experiments/smolvla_qwen_kv/evaluate_checkpoint_014000.sh
```

Run every task by calling the module without `--task-id`:

```bash
.venv/bin/python -m experiments.smolvla_qwen_kv.evaluate_checkpoint \
  --checkpoint outputs/smolvla_base_qwen_kv_all_suites/checkpoint-014000 \
  --suites libero_10 libero_spatial libero_goal libero_object \
  --episodes-per-task 20 --action-chunk 4 --save-videos --local-files-only
```

The LeRobot-integrated trainer entrypoint is:

```bash
bash experiments/smolvla_qwen_kv/run_lerobot_cached_from_014000.sh
```

This is a model warm-start. The older experiment checkpoint's
`training_state.pt` is not an Accelerate/LeRobot checkpoint tree, so its
optimizer and scheduler cannot be resumed exactly by `lerobot-train`. New
checkpoints produced by the integration use LeRobot's official layout and can
subsequently be resumed with LeRobot's standard `--resume` workflow.
