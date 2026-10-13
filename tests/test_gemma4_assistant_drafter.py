# SPDX-License-Identifier: Apache-2.0
"""Regression: a quantized Gemma 4 assistant drafter must load.

The checkpoint below ships packed int4 ``.weight`` tensors plus
``.biases``/``.scales`` sidecars; the loader must build the model quantized so
a strict ``load_weights`` binds the sidecars.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.speculative.drafters import gemma4_assistant as arch

from vllm_mlx.models.mllm import load_gemma4_assistant_drafter

DRAFTER_REPO = os.environ.get(
    "VLLM_MLX_TEST_GEMMA4_DRAFTER", "inferencerlabs/gemma-4-26B-A4B-MTP-MLX"
)


def _drafter_path() -> Path:
    from mlx_vlm.utils import get_model_path

    try:
        path = Path(get_model_path(DRAFTER_REPO))
    except Exception as exc:  # offline / not cached
        pytest.skip(f"drafter checkpoint not cached: {exc}")
    if not (path / "model.safetensors").exists():
        pytest.skip("drafter checkpoint not cached")
    return path


def _tiny_config(quantization: dict | None) -> dict:
    """A self-contained 1-layer drafter config (no checkpoint download).

    Dimensions are 64-aligned so MLX affine groups divide exactly. This
    exercises the loader's quantization branches; the real checkpoint is
    covered separately by the slow test below.
    """
    config = {
        "model_type": "gemma4_assistant",
        "architectures": ["Gemma4AssistantForCausalLM"],
        "backbone_hidden_size": 512,
        "tie_word_embeddings": True,
        "use_ordered_embeddings": False,
        "text_config": {
            "model_type": "gemma4",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
        },
    }
    if quantization is not None:
        config["quantization_config"] = quantization
    return config


def _write_checkpoint(tmp_path: Path, config: dict, quantization: dict | None):
    """Materialize a checkpoint, quantizing it exactly as the declared config says."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = arch.Model(arch.ModelConfig.from_dict(config))
    if quantization is not None:
        group_size = quantization.get("group_size", 64)
        bits = quantization.get("bits", 4)
        mode = quantization.get("mode", "affine")

        def predicate(path, module):
            override = quantization.get(path)
            if isinstance(override, dict):
                return override
            return hasattr(module, "to_quantized")

        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            mode=mode,
            class_predicate=predicate,
        )
    flat = tree_flatten(model.parameters())
    mx.eval([value for _, value in flat])
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(flat))
    return tmp_path


def test_dense_checkpoint_still_loads_unquantized(tmp_path) -> None:
    """A checkpoint with no quantization block must not be quantized."""
    config = _tiny_config(None)
    path = _write_checkpoint(tmp_path / "dense", config, None)

    model = load_gemma4_assistant_drafter(str(path))

    assert type(model.model.embed_tokens) is nn.Embedding
    assert type(model.pre_projection) is nn.Linear


def test_per_layer_override_is_honored(tmp_path) -> None:
    """A per-layer bits override in the config must reach nn.quantize."""
    quantization = {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        "pre_projection": {"group_size": 64, "bits": 8},
    }
    config = _tiny_config(quantization)
    path = _write_checkpoint(tmp_path / "mixed", config, quantization)

    model = load_gemma4_assistant_drafter(str(path))

    assert model.pre_projection.bits == 8
    assert model.model.layers[0].self_attn.q_proj.bits == 4


@pytest.mark.slow
def test_int4_drafter_loads_quantized() -> None:
    model = load_gemma4_assistant_drafter(str(_drafter_path()))

    # Sidecars bound to quantized modules; the drafter is genuinely quantized,
    # not silently dequantized. The persisted per-layer scalar must survive
    # (the model default is 1.0).
    assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(model.pre_projection, nn.QuantizedLinear)
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    embed = model.model.embed_tokens
    assert embed.scales.shape == embed.biases.shape
    assert embed.weight.shape == (262144, 128) and embed.scales.shape == (262144, 16)
    assert not mx.array_equal(
        model.model.layers[0].layer_scalar,
        mx.ones_like(model.model.layers[0].layer_scalar),
    )
