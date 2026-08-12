"""Run LeRobot's official trainer with the cached LIBERO iterable factory.

This is intentionally a thin integration wrapper.  Optimizer construction,
Accelerate/DDP, logging, checkpoint layout, WandB, and policy factories remain
LeRobot's implementation.
"""

from __future__ import annotations


def main() -> None:
    import lerobot.datasets.factory as dataset_factory

    from .lerobot_integration import make_cached_train_eval_datasets

    native_factory = dataset_factory.make_train_eval_datasets

    def integrated_factory(cfg):
        cached = make_cached_train_eval_datasets(cfg)
        return native_factory(cfg) if cached is None else cached

    integrated_factory._smolvla_qwen_cached_factory = True

    # lerobot_train imports this function into module scope. Patch before that
    # import so the rest of the upstream script remains unchanged.
    dataset_factory.make_train_eval_datasets = integrated_factory
    from lerobot.scripts import lerobot_train

    lerobot_train.make_train_eval_datasets = integrated_factory
    lerobot_train.main()


if __name__ == "__main__":
    main()
