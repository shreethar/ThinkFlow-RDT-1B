"""ThinkFlow RDT-1B LoRA training package."""

from .config import ExperimentConfig, load_config
from .data import CachedFeatureDataset, RDTBatchCollator

__all__ = [
    "ExperimentConfig",
    "load_config",
    "CachedFeatureDataset",
    "RDTBatchCollator",
    "SFTConditionedRDT",
]


def __getattr__(name: str):
    if name == "SFTConditionedRDT":
        # Keep model/PEFT imports lazy so data-only utilities can import
        # thinkflow_rdt.data even on machines with partial training deps.
        from .model import SFTConditionedRDT

        return SFTConditionedRDT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
