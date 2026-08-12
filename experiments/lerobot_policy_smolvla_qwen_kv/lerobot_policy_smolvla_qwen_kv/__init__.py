"""LeRobot third-party registration for Qwen-KV SmolVLA.

The distribution name starts with ``lerobot_policy_`` so unmodified LeRobot
entrypoints discover it through ``register_third_party_plugins()``.
"""

from __future__ import annotations

import sys
import os
import argparse
import gc
import json
import logging
from pathlib import Path


# Editable installs keep this file inside the repository.  Add the repository
# root so the implementation can remain in the isolated experiment package.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# LeRobot's default batch converter keeps a conservative fixed list of metadata
# keys.  Register qwen_kv as complementary data so the official processor
# pipeline carries it through tokenization, normalization, and device transfer.
from lerobot.processor import converters as _converters

if "qwen_kv" not in _converters._COMPLEMENTARY_KEYS:
    _converters._COMPLEMENTARY_KEYS = (*_converters._COMPLEMENTARY_KEYS, "qwen_kv")


def _disable_unused_broken_deepspeed_probe() -> None:
    """Avoid importing an incompatible optional DeepSpeed during model unwrap.

    Accelerate 1.14 checks whether the *package* is installed inside every
    ``unwrap_model`` call and imports it even for an ordinary single-GPU run.
    This environment has DeepSpeed 0.14.2, which cannot import against PyTorch
    2.10 because it references the removed ``_get_socket_with_port`` symbol.
    The custom run does not request DeepSpeed, so report it unavailable only to
    that optional unwrap probe.  An explicitly requested DeepSpeed run still
    gets the real import/error rather than being silently changed.
    """

    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true":
        return
    import accelerate.utils.other as accelerate_other

    accelerate_other.is_deepspeed_available = lambda: False


_disable_unused_broken_deepspeed_probe()


def _install_cached_dataset_factory() -> None:
    """Teach the literal ``lerobot-train`` entrypoint about cached shards.

    LeRobot has a third-party policy registry but no corresponding dataset
    registry.  Install an in-memory delegating factory: cached dataset IDs go
    to our IterableDataset and every normal LeRobot repo_id goes unchanged to
    the upstream factory.  No installed LeRobot source file is modified.
    """

    import lerobot.datasets.factory as dataset_factory

    from experiments.smolvla_qwen_kv.lerobot_integration import (
        make_cached_train_eval_datasets,
    )

    active_factory = dataset_factory.make_train_eval_datasets
    if getattr(active_factory, "_smolvla_qwen_cached_factory", False):
        integrated_factory = active_factory
    else:
        native_factory = active_factory

        def integrated_factory(cfg):
            cached = make_cached_train_eval_datasets(cfg)
            return native_factory(cfg) if cached is None else cached

        integrated_factory._smolvla_qwen_cached_factory = True
        dataset_factory.make_train_eval_datasets = integrated_factory

    # lerobot_train imports the factory into module scope before it discovers
    # plugins. Replace that already-bound reference as well.
    train_module = sys.modules.get("lerobot.scripts.lerobot_train")
    if train_module is not None:
        train_module.make_train_eval_datasets = integrated_factory


_install_cached_dataset_factory()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _comma_values(name: str, default: str) -> list[str]:
    return [value.strip() for value in os.environ.get(name, default).split(",") if value.strip()]


def _run_periodic_qwen_rollout(
    *,
    checkpoint_dir: Path,
    step: int,
    cfg,
    policy,
    preprocessor,
    postprocessor,
) -> dict:
    """Run the standalone evaluator against the trainer's in-memory policy."""

    from experiments.smolvla_qwen_kv.evaluate_checkpoint import run_evaluation
    from experiments.smolvla_qwen_kv.lerobot_integration import suites_from_repo_id

    configured_suites = _comma_values("SMOLVLA_QWEN_EVAL_SUITES", "")
    if configured_suites:
        suites = configured_suites
    else:
        suites = suites_from_repo_id(cfg.dataset.repo_id)
        if suites is None:
            raise ValueError(
                "Qwen-aware periodic rollout requires a cached_libero_qwen dataset repo_id"
            )
    task_ids = [int(value) for value in _comma_values("SMOLVLA_QWEN_EVAL_TASK_IDS", "0")]
    max_steps_value = os.environ.get("SMOLVLA_QWEN_EVAL_MAX_STEPS")
    qwen_model_id = os.environ.get(
        "SMOLVLA_QWEN_EVAL_QWEN_MODEL", "shreethar/stage1_unsloth"
    )
    args = argparse.Namespace(
        checkpoint=checkpoint_dir / "pretrained_model" / "train_config.json",
        cache_root=Path(cfg.dataset.root),
        libero_root=Path(os.environ.get("SMOLVLA_QWEN_EVAL_LIBERO_ROOT", "/home/ubuntu/LIBERO")),
        output_dir=Path(cfg.output_dir) / "qwen_rollouts" / f"step_{step:06d}",
        suites=suites,
        task_id=task_ids,
        episodes_per_task=int(os.environ.get("SMOLVLA_QWEN_EVAL_EPISODES_PER_TASK", "2")),
        env_batch_size=int(os.environ.get("SMOLVLA_QWEN_EVAL_ENV_BATCH_SIZE", "2")),
        action_chunk=int(
            os.environ.get(
                "SMOLVLA_QWEN_EVAL_ACTION_CHUNK",
                str(getattr(policy.config, "n_action_steps", 4)),
            )
        ),
        max_steps=None if max_steps_value is None else int(max_steps_value),
        seed=int(cfg.seed if cfg.seed is not None else 42),
        device=str(policy.config.device),
        qwen_device_map=os.environ.get("SMOLVLA_QWEN_EVAL_QWEN_DEVICE_MAP", "cuda"),
        qwen_model_id=qwen_model_id,
        qwen_processor_id=os.environ.get(
            "SMOLVLA_QWEN_EVAL_QWEN_PROCESSOR", qwen_model_id
        ),
        qwen_max_new_tokens=int(
            os.environ.get("SMOLVLA_QWEN_EVAL_QWEN_MAX_NEW_TOKENS", "128")
        ),
        save_videos=_env_bool("SMOLVLA_QWEN_EVAL_SAVE_VIDEOS", True),
        video_resolution=int(os.environ.get("SMOLVLA_QWEN_EVAL_VIDEO_RESOLUTION", "512")),
        video_fps=int(os.environ.get("SMOLVLA_QWEN_EVAL_VIDEO_FPS", "20")),
        local_files_only=_env_bool(
            "SMOLVLA_QWEN_EVAL_LOCAL_FILES_ONLY",
            _env_bool("HF_HUB_OFFLINE", False),
        ),
    )
    was_training = policy.training
    try:
        summary = run_evaluation(
            args,
            policy_override=policy,
            preprocessor_override=preprocessor,
            postprocessor_override=postprocessor,
        )
    finally:
        policy.reset()
        policy.train(was_training)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logging.exception("Could not clear CUDA cache after Qwen-aware rollout")

    metrics = {
        "eval/qwen_libero/success_rate": float(summary["success_rate"]),
        "eval/qwen_libero/successes": int(summary["successes"]),
        "eval/qwen_libero/episodes": int(summary["episodes"]),
    }
    for suite in summary.get("suites", []):
        metrics[f"eval/qwen_libero/{suite['suite']}/success_rate"] = float(
            suite["success_rate"]
        )
    metrics_path = Path(args.output_dir) / "wandb_metrics.json"
    metrics_path.write_text(json.dumps({"step": step, **metrics}, indent=2) + "\n")
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        logging.exception("Could not log Qwen-aware rollout metrics to WandB")
    return summary


def _install_periodic_qwen_rollout_callback() -> None:
    """Wrap LeRobot checkpoint saving with an optional live rollout callback."""

    train_module = sys.modules.get("lerobot.scripts.lerobot_train")
    if train_module is None:
        return
    active_save = train_module.save_checkpoint
    if getattr(active_save, "_smolvla_qwen_rollout_callback", False):
        return
    native_save = active_save

    def save_with_qwen_rollout(*args, **kwargs):
        native_save(*args, **kwargs)
        cfg = kwargs.get("cfg")
        step = kwargs.get("step")
        policy = kwargs.get("policy")
        checkpoint_dir = kwargs.get("checkpoint_dir")
        if cfg is None or step is None or policy is None or checkpoint_dir is None:
            return
        frequency = int(getattr(cfg, "env_eval_freq", 0))
        is_custom_policy = getattr(getattr(policy, "config", None), "type", None) == "smolvla_qwen_kv"
        is_cached_dataset = str(getattr(cfg.dataset, "repo_id", "")).startswith(
            "cached_libero_qwen"
        )
        enabled = _env_bool("SMOLVLA_QWEN_EVAL_ENABLE", True)
        if not (enabled and frequency > 0 and step % frequency == 0):
            return
        if not (is_custom_policy and is_cached_dataset):
            return
        logging.info(
            "Starting Qwen-aware LIBERO rollout at training step %s from %s",
            step,
            checkpoint_dir,
        )
        try:
            summary = _run_periodic_qwen_rollout(
                checkpoint_dir=Path(checkpoint_dir),
                step=int(step),
                cfg=cfg,
                policy=policy,
                preprocessor=kwargs.get("preprocessor"),
                postprocessor=kwargs.get("postprocessor"),
            )
            logging.info(
                "Qwen-aware rollout step=%s success=%s/%s rate=%.3f",
                step,
                summary["successes"],
                summary["episodes"],
                summary["success_rate"],
            )
        except Exception:
            logging.exception("Qwen-aware LIBERO rollout failed at step %s", step)
            if not _env_bool("SMOLVLA_QWEN_EVAL_FAIL_OPEN", True):
                raise

    save_with_qwen_rollout._smolvla_qwen_rollout_callback = True
    train_module.save_checkpoint = save_with_qwen_rollout


_install_periodic_qwen_rollout_callback()

from .configuration_smolvla_qwen_kv import KVSmolVLAConfig

__all__ = ["KVSmolVLAConfig"]
