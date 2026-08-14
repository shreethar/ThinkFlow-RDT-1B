# QwenVLA architecture notes

This directory is the isolated workspace for replacing SmolVLA's SmolVLM2
backbone with Qwen3.5 4B VLM. Nothing outside `experiments/QwenVLA` should be
modified for this experiment. Existing repository code may be imported or
subclassed read-only.

## Executive summary

SmolVLA is not simply `SmolVLM2 -> one feature vector -> action model`. It is a
paired transformer with two token streams:

```text
images -> SigLIP vision encoder -> SmolVLM connector --+
language tokens -> SmolVLM token embedding ------------+--> PREFIX (width 960)
32D padded state -> learned state_proj -----------------+
                                                        |
                                                        | layer-wise K/V context
                                                        v
Gaussian/noised 32D action chunk -> action_in_proj --> ACTION EXPERT (width 720)
flow timestep -> sinusoidal embedding -> MLP -------->  |
                                                        v
                                             action_out_proj -> 32D velocity
```

For `lerobot/smolvla_base`, both streams run through 16 transformer layers.
Even-numbered layers (0, 2, ...) perform joint masked self-attention over the
prefix and action streams. Odd-numbered layers perform prefix self-attention
plus action-expert cross-attention to the prefix K/V. Therefore images,
language, and robot state condition the action expert repeatedly at every
layer; they are not pooled into a single final embedding.

The action expert directly receives only the current noisy action chunk and
the flow-matching timestep. Images, language, and state enter the prefix/VLM
stream. Their route into the action expert is attention.

## 1. Batch fields in this repository

The current cached LIBERO experiment materializes this native SmolVLA example:

| Batch field | Shape here | Consumer |
|---|---:|---|
| `observation.images.image` | `[B,3,H,W]` | SmolVLM2 vision encoder |
| `observation.images.image2` | `[B,3,H,W]` | SmolVLM2 vision encoder |
| `observation.state` | `[B,8]`, padded to 32 | `state_proj`, then VLM prefix |
| language `task` | strings | processor -> language token IDs/mask -> VLM prefix |
| `action` | `[B,50,7]`, padded to 32 | flow target and noised action expert input |
| `action_is_pad` | `[B,50]` | loss masking only |
| `qwen_kv` (custom experiment) | `[B,T,2048]`, normally `T=1` | custom odd-layer expert cross-attention |

The local cache reader uses the current agent-view and wrist-view frames (cache
slots 3 and 4). It converts legacy RDT 11D/10D state/action representations
back to native LIBERO 8D/7D, then SmolVLA pads both to its fixed internal 32D
width. See `../smolvla_qwen_kv/cached_libero.py`.

The official SmolVLA processor:

1. adds a batch dimension when required;
2. appends a newline to the task string;
3. tokenizes it with the tokenizer named by `vlm_model_name` (48-token limit in
   the base configuration);
4. mean/std normalizes state and actions;
5. leaves visual values under identity dataset normalization and moves tensors
   to the configured device.

Inside the policy, images are aspect-preserving resized/padded to 512x512 and
mapped from `[0,1]` to `[-1,1]` for the SmolVLM2 SigLIP encoder.

## 2. What goes into SmolVLM2

`VLAFlowMatching.embed_prefix()` creates one prefix sequence in this order:

```text
[camera-1 visual tokens]
[camera-2 visual tokens]
[language token embeddings]
[one projected robot-state token]
```

For every image, SmolVLA calls SmolVLM2's vision model and then its multimodal
connector. Language IDs use SmolVLM2's ordinary text embedding table. The
state does not pass through the tokenizer: a learned `Linear(32, 960)` creates
a token in SmolVLM2 text-hidden space. Image and language embeddings are
scaled by the square root of their embedding width before concatenation.

With the base checkpoint:

- only the first 16 of SmolVLM2's 32 text layers are retained;
- text hidden width is 960;
- attention uses 15 query heads, 5 KV heads, and head width 64;
- one layer's prefix K or V tensor therefore has width `5 * 64 = 320` per
  prefix token.

The VLM LM head is irrelevant: SmolVLA never asks SmolVLM2 to generate text.
It uses the vision encoder, connector, token embeddings, and transformer
layers as a multimodal prefix network.

## 3. What goes directly into the action expert

The expert input is a suffix of 50 action tokens. It contains no raw image,
language ID, or state vector.

During training, for normalized padded action chunk `a`, sampled Gaussian
noise `eps`, and sampled scalar time `t`:

```text
x_t    = t * eps + (1 - t) * a
target = eps - a

action_emb = Linear(32, 720)(x_t)
time_emb   = sinusoidal_embedding(t, width=720), repeated for all 50 tokens
suffix     = MLP(concat(action_emb, time_emb))
```

The paired transformer predicts a velocity for each action token. A final
`Linear(720, 32)` produces `v_t`, trained by elementwise MSE against
`eps - a`. Loss is cropped to the real action dimension (7 here) and invalid
future steps are removed using `action_is_pad`.

At inference, `a` is unavailable. The suffix starts from Gaussian noise and
SmolVLA performs 10 Euler flow-integration steps from `t=1` to `t=0`. Each step
re-embeds the current noisy action chunk and timestep, runs the action expert,
and updates the action chunk.

## 4. How SmolVLM2 information reaches the action expert

The base checkpoint uses `attention_mode="cross_attn"` and
`self_attn_every_n_layers=2`, producing two alternating layer types.

### Even layers: joint masked self-attention

The VLM prefix and expert suffix use their own layer norms and Q/K/V
projections, then their Q/K/V tensors are concatenated along the token axis and
one attention operation is run.

The block mask gives these dependencies:

```text
image/language queries -> image/language keys (bidirectional prefix block)
state query            -> image/language + state keys
action query i         -> all prefix keys + action keys 0..i
prefix queries         -X-> action keys
```

Thus the VLM stream cannot leak the training action target into its prefix
representations, while the action expert can read the whole observation and
causally earlier action tokens.

### Odd layers: expert-to-VLM cross-attention

The VLM prefix performs normal prefix self-attention and generates per-layer
K/V. The expert creates queries from its current action hidden states. Before
attention, the VLM's flattened 320-wide K and V vectors are separately mapped
through the expert layer's learned K/V projections into the expert attention
space. The expert then attends only to all valid prefix tokens at that layer.

After either layer type, each stream uses its own output projection, residual,
post-attention norm, MLP, and second residual. This is why a full backbone swap
is a layer-level integration problem, not merely changing the vision encoder.

### Training versus inference caching

Training runs prefix and suffix together without a reusable cache. During
inference, SmolVLA first runs one prefix-only pass and stores every layer's
post-RoPE prefix K/V. Those fixed layer-specific K/V tensors are reused for all
10 denoising steps. Only the action suffix is recomputed at each step. At
self-attention layers, temporary suffix K/V appended during a denoising step is
cropped back to the prefix length before the next step.

## 5. What the current Qwen-KV experiment changes

The existing `../smolvla_qwen_kv` experiment does **not** replace SmolVLM2.
All native SmolVLM2 image/language/state processing above remains active.

Its cached Qwen tensor is formed from one selected Qwen sequence position at
one selected Qwen layer:

```text
Qwen key   [KV heads, head width] -> flatten -> 1024
Qwen value [KV heads, head width] -> flatten -> 1024
concatenate                                    2048
```

The position is normally immediately before `</think>` (or the final response
terminator fallback). Qwen itself receives the configured image list plus a
trajectory prompt derived from the task instruction. See
`../../scripts/precompute_all_features.py`.

At each odd SmolVLA cross-attention layer, separate learned, layer-specific
adapters map Qwen K and V from 1024 values to SmolVLM2's native 320-wide source
KV shape (`5 heads * 64`). The resulting token is appended to that layer's
ordinary SmolVLM2 prefix K/V. A learned per-layer/per-query-head logit bias,
initialized to -4, controls how quickly the expert starts using it.

So the current expert context is:

```text
[all SmolVLM2 image/language/state prefix tokens] + [one adapted Qwen token]
```

It is not Qwen-only conditioning. The appended token is also used only at the
odd cross-attention layers; it is not part of the alternating even-layer joint
self-attention stream.

## 6. Correct replacement boundary for full QwenVLA

The narrow class to replace is `SmolVLMWithExpertModel`, together with the
Smol-specific parts of `VLAFlowMatching.embed_prefix()`. The reusable pieces
are the action preprocessing, flow-matching objective/integrator, 32D action
projections, action expert weights, loss masking, action queue, and LeRobot
pre/post-processing contract.

A faithful full replacement needs to produce **layer-specific, multi-token
Qwen prefix context**, rather than project the single `[B,1,2048]` summary into
every expert layer. The intended conceptual contract is:

```text
Qwen vision + language + injected robot state
    -> Qwen layer 0 prefix K/V  -> expert fusion layer 0
    -> Qwen layer 1 prefix K/V  -> expert fusion layer 1
    ...
    -> selected/mapped Qwen layer K/V -> corresponding expert layer
```

Important incompatibilities that must be handled explicitly:

- Qwen and SmolVLM2 have different hidden widths, KV layouts, layer counts,
  rotary-position schemes, processor outputs, vision connectors, and internal
  module names.
- The pretrained action expert is 720-wide and its cross-attention expects a
  320-wide SmolVLM source K/V input. Preserving the expert therefore requires
  layer-specific Qwen-to-320 K/V adapters (or a deliberate expert weight
  reinitialization).
- The current `state_proj` emits 960-wide SmolVLM tokens. A Qwen prefix needs a
  new state projection into Qwen hidden width and a valid Qwen position/modality
  treatment. Dropping state would change the policy's information contract.
- SmolVLA relies on the prefix being independent of the action target so it can
  cache it once at inference. That separation must remain intact.
- Replacing the backbone invalidates direct loading of SmolVLM2 prefix weights,
  but does not inherently require discarding the trained flow/action expert.
- A 4B Qwen prefix is likely to dominate memory and latency unless its prefix
  pass is cached once per observation and reused across all denoising steps.

## 7. Implementation checkpoints for the next phase

Before training, the safest progression is:

1. Build a Qwen prefix adapter that returns per-layer K/V plus masks without an
   action expert and validate them against a normal Qwen forward pass.
2. Freeze Qwen and the pretrained SmolVLA expert; train only per-layer K/V
   adapters and the new state projection on a tiny overfit subset.
3. Verify that shuffling images, language, state, and Qwen layer contexts each
   changes predicted action chunks. This catches an apparently training model
   that ignores its condition.
4. Verify prefix K/V is computed once per observation during inference, not
   once per Euler step.
5. Only then consider unfreezing Qwen or the expert.

## Source map

Local read-only sources used for this trace:

- `../smolvla_qwen_kv/modeling.py`: current Qwen-KV fusion implementation.
- `../smolvla_qwen_kv/cached_libero.py`: exact cached batch schema.
- `../smolvla_qwen_kv/configuration.py`: current LIBERO and external-KV config.
- `../smolvla_qwen_kv/train_cached.py`: processor and training path.
- `../../scripts/precompute_all_features.py`: Qwen KV extraction semantics.

Official upstream references inspected on 2026-08-14:

- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/smolvlm_with_expert.py
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/processor_smolvla.py
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/common/vla_utils.py
- https://huggingface.co/lerobot/smolvla_base/blob/d5ef92b547b2bf36bdd50f18ea6ed6463cb5c5af/config.json
