"""Small tests that do not allocate the full SmolVLA checkpoint."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .cached_libero import (
    _pad_action_chunk,
    cached_action_to_libero_action,
    cached_state_to_libero_state,
)
from .modeling import KVSmolVLAPolicy, eager_grouped_query_attention


def test_grouped_query_attention_shape_and_finiteness() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(2, 4, 6, 8, generator=generator)
    key = torch.randn(2, 5, 2, 8, generator=generator)
    value = torch.randn(2, 5, 2, 8, generator=generator)
    mask = torch.ones(2, 4, 5, dtype=torch.bool)
    output = eager_grouped_query_attention(mask, query, key, value)
    assert output.shape == (2, 4, 48)
    assert torch.isfinite(output).all()


def test_appended_key_is_not_degenerate_because_it_competes_with_prefix() -> None:
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(1, 3, 4, 8, generator=generator)
    prefix_key = torch.randn(1, 2, 2, 8, generator=generator)
    external_key = torch.randn(1, 1, 2, 8, generator=generator)
    prefix_value = torch.randn(1, 2, 2, 8, generator=generator)
    external_value = torch.randn(1, 1, 2, 8, generator=generator)
    key = torch.cat([prefix_key, external_key], dim=1)
    value = torch.cat([prefix_value, external_value], dim=1)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    bias = torch.zeros(4)
    first = eager_grouped_query_attention(
        mask,
        query,
        key,
        value,
        appended_token_count=1,
        appended_logit_bias=bias,
    )
    key[:, -1].add_(3.0)
    second = eager_grouped_query_attention(
        mask,
        query,
        key,
        value,
        appended_token_count=1,
        appended_logit_bias=bias,
    )
    assert not torch.allclose(first, second)


def test_single_replacement_token_would_make_key_irrelevant() -> None:
    generator = torch.Generator().manual_seed(19)
    query = torch.randn(1, 3, 4, 8, generator=generator)
    key = torch.randn(1, 1, 2, 8, generator=generator)
    value = torch.randn(1, 1, 2, 8, generator=generator)
    mask = torch.ones(1, 3, 1, dtype=torch.bool)
    first = eager_grouped_query_attention(mask, query, key, value)
    second = eager_grouped_query_attention(mask, query * 100, key * -100, value)
    assert torch.allclose(first, second)


def test_policy_accepts_initialized_external_token_count() -> None:
    policy = SimpleNamespace(
        config=SimpleNamespace(
            external_kv_key="qwen_kv",
            external_kv_required=True,
            external_kv_width=2048,
            external_kv_token_count=5,
        )
    )
    external = torch.zeros(2, 5, 2048)
    result = KVSmolVLAPolicy._external_kv_from_batch(policy, {"qwen_kv": external})
    assert result is external


def test_policy_rejects_cache_from_other_external_token_count() -> None:
    policy = SimpleNamespace(
        config=SimpleNamespace(
            external_kv_key="qwen_kv",
            external_kv_required=True,
            external_kv_width=2048,
            external_kv_token_count=5,
        )
    )
    with pytest.raises(ValueError, match="Expected 5 Qwen KV tokens"):
        KVSmolVLAPolicy._external_kv_from_batch(
            policy,
            {"qwen_kv": torch.zeros(2, 1, 2048)},
        )


def test_action_chunk_padding_and_validity() -> None:
    actions = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    valid = torch.tensor([True, True, False])
    padded, padded_valid = _pad_action_chunk(actions, valid, 5)
    assert padded.shape == (5, 7)
    assert padded_valid.tolist() == [True, True, False, False, False]
    assert torch.equal(padded[:3], actions)
    assert not padded[3:].any()


def _rotvec_to_ortho6d(rotvec: np.ndarray) -> np.ndarray:
    matrix = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    matrix = matrix.reshape(*rotvec.shape[:-1], 3, 3)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def test_cached_state_decodes_to_native_libero_8d() -> None:
    xyz = np.array([[0.1, -0.2, 0.9]], dtype=np.float32)
    rotvec = np.array([[2.9, 0.1, -0.2]], dtype=np.float32)
    fingers = np.array([[0.035, -0.035]], dtype=np.float32)
    cached = np.concatenate([xyz, _rotvec_to_ortho6d(rotvec), fingers], axis=-1)
    decoded = cached_state_to_libero_state(cached).numpy()
    np.testing.assert_allclose(decoded[..., :3], xyz, atol=1e-6)
    np.testing.assert_allclose(decoded[..., 3:6], rotvec, atol=1e-6)
    np.testing.assert_allclose(decoded[..., 6:8], fingers, atol=1e-6)


def test_cached_action_decodes_to_raw_libero_7d() -> None:
    raw = np.array(
        [[0.2, -0.3, 0.4, 0.7, -0.6, 0.2, -1.0]],
        dtype=np.float32,
    )
    relative_ortho6d = _rotvec_to_ortho6d(raw[..., 3:6] * 0.5)
    cached = np.concatenate([raw[..., :3], relative_ortho6d, raw[..., 6:7]], axis=-1)
    decoded = cached_action_to_libero_action(cached).numpy()
    np.testing.assert_allclose(decoded, raw, atol=1e-6)
