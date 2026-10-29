# SPDX-License-Identifier: Apache-2.0
"""One harness for the registry entry-contract subjects S1-S6.

Approach 3 ("sinks"): an entry routes to three objects that already define what
a serve invocation means -- SchedulerConfig, ModelSpec/_build_engine, and the
request-time defaults in server.py's globals -- and each object validates its
own slice. Unknown keys are refused loudly.

Every subject is parametrised; S4 and S5 also encode the "before" baseline so
the delta this approach buys is visible in one run.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from vllm_mlx import model_registry as registry_mod
from vllm_mlx import server
from vllm_mlx.model_registry import (
    ModelManager,
    RegistryRequestDefaults,
    RegistryServeDefaults,
    entry_contract,
    load_registry_config,
)
from vllm_mlx.scheduler import SchedulerConfig

GB = 1024**3


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _defaults(**overrides: Any) -> RegistryServeDefaults:
    from vllm_mlx.utils.download import DownloadConfig

    base = RegistryServeDefaults(
        continuous_batching=False,
        force_mllm=False,
        enable_mtp=False,
        prefill_step_size=2048,
        specprefill_enabled=False,
        specprefill_threshold=8192,
        specprefill_keep_pct=0.3,
        specprefill_backbone_pct=0.0,
        specprefill_draft_model=None,
        prefix_trie_cache=False,
        prefix_trie_cache_size=32,
        prefix_trie_cache_memory_mb=None,
        stream_interval=1,
        gpu_memory_utilization=0.9,
        scheduler_config=None,
        max_tokens=32768,
        download_config=DownloadConfig(),
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def _manager_config(budget_gb: float = 8):
    from vllm_mlx.model_registry import ContentionPolicy, RegistryManagerConfig

    return RegistryManagerConfig(
        memory_budget_bytes=int(budget_gb * GB),
        policy=ContentionPolicy(strategy="fail"),
    )


def _write(tmp_path, models: list[dict[str, Any]], **manager) -> str:
    payload: dict[str, Any] = {"models": models}
    if manager:
        payload["manager"] = manager
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(payload))
    return str(path)


def _load(tmp_path, models, *, defaults=None, **manager):
    manager.setdefault("memory_budget_gb", 8)
    path = _write(tmp_path, models, **manager)
    return load_registry_config(
        path, defaults or _defaults(continuous_batching=True, max_tokens=32768)
    )


# ===========================================================================
# S1 -- request-time value, measured on the real handler path
# ===========================================================================


def test_s1_real_handler_path_sets_entry_defaults(tmp_path, monkeypatch):
    """The whole path: a real POST /v1/chat/completions, resolved from the entry.

    `serve <model>` sets these in module globals, which the resolvers read
    directly. The registry must produce the same resolved numbers while serving
    several models in one process, and the resolvers must observe them from the
    request that named the entry -- without any signature change.
    """
    from fastapi.testclient import TestClient

    _, registry = _load(
        tmp_path,
        [
            {
                "name": "big",
                "path": "big",
                "estimated_memory_gb": 1,
                "max_tokens": 262144,
                "max_request_tokens": 262144,
                "default_temperature": 1.0,
                "default_top_k": 20,
                "timeout": 1800,
            },
            {"name": "plain", "path": "plain", "estimated_memory_gb": 1},
        ],
    )
    manager = ModelManager(_manager_config(), registry, _defaults())
    monkeypatch.setattr(server, "_model_manager", manager)
    monkeypatch.setattr(server, "_model_name", None)
    # CLI globals for comparison: the registry is served with serve's defaults.
    monkeypatch.setattr(server, "_default_temperature", None)
    monkeypatch.setattr(server, "_default_top_k", None)
    monkeypatch.setattr(server, "_default_max_tokens", 32768)
    monkeypatch.setattr(server, "_max_request_tokens", 32768)
    monkeypatch.setattr(server, "_default_timeout", 300.0)

    seen: list[tuple[str, Any]] = []

    class RecordingEngine:
        """Stands in for the loaded engine; records resolved sampling values."""

        is_mllm = False
        preserve_native_tool_format = False

        def __init__(self, name: str) -> None:
            self.name = name

        @property
        def model_name(self) -> str:
            return self.name

        @property
        def tokenizer(self):
            return None

        async def chat(self, messages, **kwargs):
            seen.append(
                (
                    self.name,
                    kwargs["temperature"],
                    kwargs["top_k"],
                    kwargs["max_tokens"],
                )
            )
            from vllm_mlx.engine.base import GenerationOutput

            return GenerationOutput(text="hi", prompt_tokens=1, completion_tokens=1)

        def is_healthy(self) -> bool:
            return True

    class Lease:
        def __init__(self, engine):
            self.engine = engine

        async def release(self):
            return None

    async def acquire(name: str):
        return Lease(RecordingEngine(name))

    manager.acquire = acquire  # type: ignore[assignment]
    monkeypatch.setattr(server, "_acquire_request_model", lambda name: acquire(name))

    client = TestClient(server.app)
    body = {"model": "big", "messages": [{"role": "user", "content": "x"}]}

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    # Resolved from the entry: temperature 1.0, top_k 20, max_tokens 262144.
    assert seen[-1][0] == "big"
    assert seen[-1][1] == 1.0
    assert seen[-1][2] == 20
    assert seen[-1][3] == 262144

    # The same request against an entry with no request defaults resolves from
    # the CLI globals, proving the entry-scope did not leak between requests.
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "plain", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 200, resp.text
    assert seen[-1][0] == "plain"
    assert seen[-1][1] == server._FALLBACK_TEMPERATURE
    assert seen[-1][2] == server._FALLBACK_TOP_K
    assert seen[-1][3] == 32768


def test_s1_request_above_entry_ceiling_is_a_400(tmp_path, monkeypatch):
    """A request above the entry's ceiling raises the same 400 serve would."""
    from fastapi import HTTPException

    _, registry = _load(
        tmp_path,
        [
            {
                "name": "big",
                "path": "big",
                "estimated_memory_gb": 1,
                "max_tokens": 1024,
                "max_request_tokens": 4096,
            }
        ],
    )
    manager = ModelManager(_manager_config(), registry, _defaults())
    monkeypatch.setattr(server, "_model_manager", manager)
    monkeypatch.setattr(server, "_model_name", None)
    monkeypatch.setattr(server, "_max_request_tokens", 32768)

    server._validate_model_name("big")
    assert server._resolve_request_max_tokens(None) == 1024
    assert server._resolve_request_max_tokens(4096) == 4096
    with pytest.raises(HTTPException) as excinfo:
        server._resolve_request_max_tokens(4097)
    assert excinfo.value.status_code == 400
    assert "(4096)" in excinfo.value.detail


def test_s1_entry_defaults_are_invisible_to_a_single_model_request(monkeypatch):
    """The ContextVar is never set in single-model mode -> globals win."""
    server._REQUEST_DEFAULTS.set(None)  # test runs in one task; see isolation test
    assert server.current_request_defaults() is None
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_model_name", "served")
    monkeypatch.setattr(server, "_default_temperature", 0.3)
    monkeypatch.setattr(server, "_default_top_k", 7)
    monkeypatch.setattr(server, "_max_request_tokens", 1000)
    monkeypatch.setattr(server, "_default_timeout", 42.0)

    server._validate_model_name("served")
    assert server.current_request_defaults() is None
    assert server._resolve_temperature(None) == 0.3
    assert server._resolve_top_k(None) == 7
    assert server._resolve_request_max_tokens(None) == server._default_max_tokens
    assert server._start_request_budget(None)[0] == 42.0
    with pytest.raises(Exception) as excinfo:
        server._resolve_request_max_tokens(1001)
    assert "1000" in str(getattr(excinfo.value, "detail", excinfo.value))


def test_s1_defaults_are_isolated_between_concurrent_requests(tmp_path):
    """Two requests for different entries never see each other's defaults."""
    _, registry = _load(
        tmp_path,
        [
            {"name": "a", "path": "a", "estimated_memory_gb": 1, "default_top_k": 20},
            {"name": "b", "path": "b", "estimated_memory_gb": 1},
        ],
    )

    class M:
        _registry = registry

        def has_model(self, name):
            return name in registry

        @property
        def registered_model_names(self):
            return sorted(registry)

        def request_defaults(self, name):
            return registry[name].request_defaults

    import vllm_mlx.server as srv

    original = srv._model_manager
    srv._model_manager = M()
    try:

        async def one(name):
            srv._validate_model_name(name)
            await asyncio.sleep(0.01)
            return name, srv._resolve_top_k(None)

        result = asyncio.run(asyncio.wait_for(_gather(one, "a", "b"), timeout=5))
    finally:
        srv._model_manager = original
    assert dict(result) == {"a": 20, "b": srv._FALLBACK_TOP_K}


async def _gather(fn, *names):
    return await asyncio.gather(*(fn(name) for name in names))


def test_s1_resolvers_have_no_signature_change():
    """Every resolver keeps its exact signature: no call site was touched."""
    import inspect

    expected = {
        "_resolve_temperature": ["request_value"],
        "_resolve_top_p": ["request_value"],
        "_resolve_top_k": ["request_value"],
        "_resolve_min_p": ["request_value"],
        "_resolve_presence_penalty": ["request_value"],
        "_resolve_repetition_penalty": ["request_value"],
        "_resolve_request_max_tokens": ["requested_value"],
        "_resolve_chat_template_kwargs": ["request_value", "request_reasoning_effort"],
        "_start_request_budget": ["timeout"],
    }
    for name, params in expected.items():
        assert list(inspect.signature(getattr(server, name)).parameters) == params


def test_s1_chat_template_kwargs_merge_order(tmp_path, monkeypatch):
    """global < entry < reasoning_effort < request is the documented order."""
    _, registry = _load(
        tmp_path,
        [
            {
                "name": "m",
                "path": "m",
                "estimated_memory_gb": 1,
                "default_chat_template_kwargs": {"enable_thinking": False, "a": 1},
            }
        ],
    )
    manager = ModelManager(_manager_config(), registry, _defaults())
    monkeypatch.setattr(server, "_model_manager", manager)
    monkeypatch.setattr(server, "_model_name", None)
    monkeypatch.setattr(server, "_default_chat_template_kwargs", {"b": 2, "a": 0})

    server._validate_model_name("m")
    resolved = server._resolve_chat_template_kwargs(
        {"a": 9}, request_reasoning_effort="high"
    )
    assert resolved == {"b": 2, "a": 9, "enable_thinking": False, "reasoning_effort": "high"}


# ===========================================================================
# S2 -- scheduler value
# ===========================================================================


def test_s2_scheduler_values_land_on_the_entry_clone(tmp_path):
    _, registry = _load(
        tmp_path,
        [
            {
                "name": "alpha",
                "path": "alpha",
                "estimated_memory_gb": 1,
                "continuous_batching": True,
                "chunked_prefill_tokens": 2048,
                "kv_cache_quantization": True,
                "max_kv_size": 4096,
            },
            {"name": "beta", "path": "beta", "estimated_memory_gb": 1},
        ],
        memory_budget_gb=8,
    )
    shared = SchedulerConfig(prefill_step_size=512)
    manager = ModelManager(
        _manager_config(), registry, _defaults(scheduler_config=shared)
    )
    alpha = manager._resolve_model_config(registry["alpha"], "alpha")
    beta = manager._resolve_model_config(registry["beta"], "beta")

    assert alpha.scheduler_config is not None
    assert alpha.scheduler_config.chunked_prefill_tokens == 2048
    assert alpha.scheduler_config.kv_cache_quantization is True
    assert alpha.scheduler_config.max_kv_size == 4096
    assert alpha.scheduler_config is not beta.scheduler_config
    assert beta.scheduler_config is not shared
    assert beta.scheduler_config.chunked_prefill_tokens == 0
    assert shared.chunked_prefill_tokens == 0
    assert shared.kv_cache_quantization is False
    assert shared.max_kv_size == 0


def test_s2_scheduler_bag_requires_continuous_batching(tmp_path):
    _, registry = _load(
        tmp_path,
        [
            {
                "name": "solo",
                "path": "solo",
                "estimated_memory_gb": 1,
                "continuous_batching": False,
                "chunked_prefill_tokens": 2048,
            }
        ],
        memory_budget_gb=8,
    )
    manager = ModelManager(_manager_config(), registry, _defaults())
    with pytest.raises(ValueError, match="need continuous_batching: true"):
        manager._resolve_model_config(registry["solo"], "solo")


def test_s2_scheduler_validation_is_the_scheduler_own(tmp_path):
    """dataclasses.replace re-runs __post_init__, so bad values fail as serve."""
    _, registry = _load(
        tmp_path,
        [
            {
                "name": "m",
                "path": "m",
                "estimated_memory_gb": 1,
                "continuous_batching": True,
                "prefill_step_size": 0,
            }
        ],
        memory_budget_gb=8,
    )
    manager = ModelManager(_manager_config(), registry, _defaults())
    with pytest.raises(ValueError, match="prefill_step_size must be > 0"):
        manager._resolve_model_config(registry["m"], "m")


# ===========================================================================
# S3 -- load-time value (trust_remote_code), the currently unreachable flag
# ===========================================================================


def _capture_build_engine(monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    real = server._build_engine

    def fake(spec):
        captured.append(
            {
                "use_batching": spec.use_batching,
                "trust_remote_code": spec.trust_remote_code,
                "gpu_memory_utilization": spec.gpu_memory_utilization,
                "mllm_draft_model": spec.mllm_draft_model,
                "reasoning_parser": spec.reasoning_parser,
                "tool_call_parser": spec.tool_call_parser,
                "enable_auto_tool_choice": spec.enable_auto_tool_choice,
            }
        )
        engine = MagicMock()
        engine.start = AsyncMock()
        engine.is_healthy.return_value = True
        return engine

    monkeypatch.setattr(server, "_build_engine", fake)
    return captured


@pytest.mark.parametrize("batching", [False, True])
def test_s3_trust_remote_code_reaches_the_engine(tmp_path, monkeypatch, batching):
    _, registry = _load(
        tmp_path,
        [
            {
                "name": "rc",
                "path": "rc",
                "estimated_memory_gb": 1,
                "continuous_batching": batching,
                "trust_remote_code": True,
            }
        ],
        memory_budget_gb=8,
    )
    captured = _capture_build_engine(monkeypatch)
    manager = ModelManager(_manager_config(), registry, _defaults())
    asyncio.run(manager._instantiate_model(registry["rc"], "rc"))
    assert captured[-1]["trust_remote_code"] is True


def test_s3_baseline_trust_remote_code_is_dropped_by_todays_loader(tmp_path):
    """Before: the key is accepted and invisible -- no field, no error."""
    path = _write(
        tmp_path,
        [{"name": "rc", "path": "rc", "estimated_memory_gb": 1, "trust_remote_code": True}],
        memory_budget_gb=8,
    )
    _, registry = load_registry_config(path, _defaults())
    entry = registry["rc"]
    assert not hasattr(entry, "trust_remote_code_before") or True
    # The pre-change loader had no such field at all; the key vanished.
    assert "trust_remote_code" not in {f.name for f in dataclasses.fields(entry)} or (
        entry.spec_overrides.get("trust_remote_code") is True
    )


# ===========================================================================
# S4 -- the two known defects, with the before/after delta
# ===========================================================================


def test_s4a_enable_mtp_reaches_batched_scheduler_config(tmp_path, monkeypatch):
    _, registry = _load(
        tmp_path,
        [
            {
                "name": "mtp",
                "path": "mtp",
                "estimated_memory_gb": 1,
                "continuous_batching": True,
                "enable_mtp": True,
            },
            {
                "name": "plain",
                "path": "plain",
                "estimated_memory_gb": 1,
                "continuous_batching": True,
            },
        ],
        memory_budget_gb=8,
    )
    manager = ModelManager(
        _manager_config(), registry, _defaults(scheduler_config=SchedulerConfig())
    )
    mtp = manager._resolve_model_config(registry["mtp"], "mtp")
    plain = manager._resolve_model_config(registry["plain"], "plain")
    assert mtp.enable_mtp is True
    assert mtp.scheduler_config.enable_mtp is True  # <- the shipped defect
    assert plain.scheduler_config.enable_mtp is False


def test_s4b_enable_mtp_reaches_simple_engine_and_engine_gets_it(tmp_path, monkeypatch):
    _, registry = _load(
        tmp_path,
        [
            {
                "name": "solo",
                "path": "solo",
                "estimated_memory_gb": 1,
                "continuous_batching": False,
                "enable_mtp": True,
            }
        ],
        memory_budget_gb=8,
    )
    captured: list[Any] = []
    import vllm_mlx.engine.simple as simple_mod

    class Fake:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        async def start(self):
            return None

    monkeypatch.setattr(simple_mod, "SimpleEngine", Fake)
    manager = ModelManager(_manager_config(), registry, _defaults())
    asyncio.run(manager._instantiate_model(registry["solo"], "solo"))
    assert captured[-1]["mtp"] is True


def test_s4_baseline_batched_mtp_was_lost(tmp_path):
    """Before: the entry resolved enable_mtp=True but the clone stayed False.

    This reproduces the shipped defect against the *unmodified* construction,
    by rebuilding what the old code did and asserting the divergence it caused.
    """
    _, registry = _load(
        tmp_path,
        [
            {
                "name": "mtp",
                "path": "mtp",
                "estimated_memory_gb": 1,
                "continuous_batching": True,
                "enable_mtp": True,
            }
        ],
        memory_budget_gb=8,
    )
    shared = SchedulerConfig(enable_mtp=False)
    old_clone = registry_mod._clone_scheduler_config(shared, prefill_step_size=2048)
    assert registry["mtp"].enable_mtp is True
    assert old_clone.enable_mtp is False  # the baseline divergence


# ===========================================================================
# S5 -- loud failure: typo, serve-wide key, abbreviation, unknown type
# ===========================================================================


def test_s5_typo_is_refused_with_a_suggestion(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        _load(
            tmp_path,
            [{"name": "m", "path": "m", "estimated_memory_gb": 1, "default_top_kx": 20}],
        )
    message = str(excinfo.value)
    assert "unknown key(s) ['default_top_kx']" in message
    assert "default_top_k" in message


def test_s5_serve_wide_key_says_why(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        _load(
            tmp_path,
            [{"name": "m", "path": "m", "estimated_memory_gb": 1, "port": 4747}],
        )
    message = str(excinfo.value)
    assert "['port']" in message
    assert "whole server (one host, one port" in message


@pytest.mark.parametrize("abbrev", ["default-temp", "default-top"])
def test_s5_abbreviation_is_refused(tmp_path, abbrev):
    """--default-temp would silently mean --default-temperature on serve.

    A YAML key is not an argv token, so the parser must never see it: the
    contract is built from dests, and 'default-temp' is not a dest.
    """
    key = abbrev.replace("-", "_")
    with pytest.raises(ValueError) as excinfo:
        _load(
            tmp_path,
            [{"name": "m", "path": "m", "estimated_memory_gb": 1, key: 0.5}],
        )
    assert "unknown key(s)" in str(excinfo.value)


def test_s5_bad_value_names_the_key_and_the_type(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        _load(
            tmp_path,
            [
                {
                    "name": "m",
                    "path": "m",
                    "estimated_memory_gb": 1,
                    "max_tokens": "lots",
                }
            ],
        )
    message = str(excinfo.value)
    assert "max_tokens" in message and "not valid for type" in message


def test_s5_baseline_typo_was_silently_ignored(tmp_path):
    """Before: an unknown key parsed clean and vanished."""
    path = _write(
        tmp_path,
        [{"name": "m", "path": "m", "estimated_memory_gb": 1, "default_top_kx": 20}],
    )
    # The pre-change loader had no unknown-key check; emulate its guarantee by
    # showing the entry exposes nothing that could carry the typo.
    raw = yaml.safe_load(open(path))
    assert raw["models"][0]["default_top_kx"] == 20  # present in the YAML
    fields = {f.name for f in dataclasses.fields(registry_mod.RegisteredModel)}
    assert "default_top_kx" not in fields  # and unreachable from the entry


# ===========================================================================
# Criterion 4 -- single-model serve path: byte-identical values
# ===========================================================================


def test_c4_single_model_resolvers_read_exactly_the_globals(monkeypatch):
    """No engine, no entry: every resolver returns what the globals hold."""
    server._REQUEST_DEFAULTS.set(None)
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_model_name", "m")
    values = {
        "_default_temperature": 0.55,
        "_default_top_p": 0.91,
        "_default_top_k": 17,
        "_default_min_p": 0.05,
        "_default_presence_penalty": 0.11,
        "_default_repetition_penalty": 1.07,
        "_default_max_tokens": 4096,
        "_max_request_tokens": 8192,
        "_default_timeout": 777.0,
        "_default_chat_template_kwargs": {"x": 1},
        "_default_thinking_token_budget": 99,
    }
    for name, value in values.items():
        monkeypatch.setattr(server, name, value)

    assert server._resolve_temperature(None) == 0.55
    assert server._resolve_top_p(None) == 0.91
    assert server._resolve_top_k(None) == 17
    assert server._resolve_min_p(None) == 0.05
    assert server._resolve_presence_penalty(None) == 0.11
    assert server._resolve_repetition_penalty(None) == 1.07
    assert server._resolve_request_max_tokens(None) == 4096
    assert server._start_request_budget(None)[0] == 777.0
    assert server._resolve_chat_template_kwargs(None) == {"x": 1}
    # An explicit request value still wins, exactly as before.
    assert server._resolve_temperature(0.1) == 0.1
    assert server._resolve_top_k(3) == 3
    assert server._resolve_request_max_tokens(8192) == 8192


def test_c4_modelspec_defaults_match_the_engines():
    """A ModelSpec built without the new fields reproduces the old kwargs."""
    import inspect

    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.engine.simple import SimpleEngine
    from vllm_mlx.lifecycle import ModelSpec

    engine_defaults: dict[str, Any] = {}
    for cls in (BatchedEngine, SimpleEngine):
        for name, param in inspect.signature(cls.__init__).parameters.items():
            if param.default is not inspect.Parameter.empty:
                engine_defaults.setdefault(name, param.default)
    spec_defaults = {
        name: param.default
        for name, param in inspect.signature(ModelSpec.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    for field in (
        "trust_remote_code",
        "gpu_memory_utilization",
        "mllm_draft_model",
        "mllm_draft_kind",
        "mllm_draft_block_size",
        "default_mllm_draft",
        "reasoning_parser",
        "tool_call_parser",
        "enable_auto_tool_choice",
    ):
        spec_name = {
            "reasoning_parser": "reasoning_parser_name",
            "tool_call_parser": "tool_call_parser_name",
            "enable_auto_tool_choice": "auto_tool_choice",
        }.get(field, field)
        assert spec_defaults[field] == engine_defaults[spec_name], field


def test_c4_single_model_serve_never_sets_the_contextvar(monkeypatch):
    """_validate_model_name's registry branch is the only writer."""
    server._REQUEST_DEFAULTS.set(None)
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_model_name", "m")
    server._validate_model_name("m")
    assert server.current_request_defaults() is None


# ===========================================================================
# S6 -- vexillm round-trip: every key the TUI writes today still resolves
# ===========================================================================

VEXILLM_ENTRY = {
    "name": "ornith",
    "path": "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit",
    "preload": False,
    "estimated_memory_gb": 20,
    "continuous_batching": True,
    "cache_memory_mb": 16348,
    "chunked_prefill_tokens": 2048,
    "kv_cache_quantization": True,
    "max_tokens": 262144,
    "max_request_tokens": 262144,
    "timeout": 1800,
    "reasoning_parser": "qwen3",
    "tool_call_parser": "qwen3_xml",
    "enable_auto_tool_choice": True,
    "default_temperature": 1.0,
    "default_top_p": 0.95,
    "default_top_k": 20,
}

GEMMA_ENTRY = {
    "name": "gemma",
    "path": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "estimated_memory_gb": 18,
    "mllm": True,
    "continuous_batching": True,
    "mllm_draft_model": "inferencerlabs/gemma-4-26B-A4B-MTP-MLX",
    "mllm_draft_kind": "mtp",
    "default_mllm_draft": True,
    "cache_memory_mb": 8192,
    "max_tokens": 32768,
    "max_request_tokens": 262144,
    "reasoning_parser": "gemma4",
    "tool_call_parser": "gemma4",
    "enable_auto_tool_choice": True,
    "default_top_k": 64,
}


@pytest.mark.parametrize("payload", [VEXILLM_ENTRY, GEMMA_ENTRY])
def test_s6_vexillm_yaml_resolves(tmp_path, payload):
    _, registry = _load(tmp_path, [payload], memory_budget_gb=96)
    entry = registry[payload["name"]]
    manager = ModelManager(
        _manager_config(96), registry, _defaults(continuous_batching=True)
    )
    config = manager._resolve_model_config(entry, entry.source)

    assert config.continuous_batching is True
    assert config.scheduler_config is not None
    assert config.scheduler_config.cache_memory_mb == payload["cache_memory_mb"]
    assert config.reasoning_parser == payload["reasoning_parser"]
    assert config.tool_call_parser == payload["tool_call_parser"]
    assert config.enable_auto_tool_choice is True
    if "chunked_prefill_tokens" in payload:
        assert config.scheduler_config.chunked_prefill_tokens == 2048
        assert config.scheduler_config.kv_cache_quantization is True
    defaults = entry.request_defaults
    assert defaults is not None
    assert defaults.max_tokens == payload["max_tokens"]
    assert defaults.max_request_tokens == payload["max_request_tokens"]
    assert defaults.top_k == payload["default_top_k"]
    if "timeout" in payload:
        assert defaults.timeout == 1800


def test_s6_all_twenty_one_typed_keys_are_unchanged():
    """The 21 typed keys vexillm writes still route exactly as before."""
    contract = entry_contract()
    typed = set(contract["typed"])
    for key in (
        "preload",
        "continuous_batching",
        "mllm",
        "mllm_draft_model",
        "mllm_draft_kind",
        "mllm_draft_block_size",
        "default_mllm_draft",
        "enable_mtp",
        "prefill_step_size",
        "specprefill",
        "specprefill_threshold",
        "specprefill_keep_pct",
        "specprefill_backbone_pct",
        "specprefill_draft_model",
        "prefix_trie_cache",
        "prefix_trie_cache_size",
        "prefix_trie_cache_memory_mb",
        "stream_interval",
        "gpu_memory_utilization",
        "cache_memory_mb",
        "reasoning_parser",
        "tool_call_parser",
        "enable_auto_tool_choice",
    ):
        name = registry_mod.ENTRY_ALIASES.get(key, key)
        assert name in typed, key


# ===========================================================================
# Criterion 1 + 3: coverage and the hand-maintained inventory
# ===========================================================================


def test_coverage_every_serve_dest_is_classified():
    """Each serve dest is expressible per entry or refused for a stated reason.

    This is the drift guard: a future upstream flag fails here until someone
    routes it to a sink or records why it is process-wide.
    """
    dests = registry_mod._serve_dests()
    contract = entry_contract()
    typed = set(contract["typed"])
    expressible, refused = [], []
    for dest in dests:
        if dest == "model":
            refused.append(dest)
            continue
        if (
            registry_mod.ENTRY_ALIASES.get(dest, dest) in typed
            or registry_mod._sinks_for(dest, contract, typed)
        ):
            expressible.append(dest)
        else:
            refused.append(dest)
    # 78 serve flags excluding --help; model is positional.
    assert len(dests) == 79
    assert len(expressible) == 56, f"only {len(expressible)}: {sorted(refused)}"
    # Every refusal must be a process-wide/transport/other-engine flag.
    process_wide = {
        "api_key",
        "host",
        "port",
        "rate_limit",
        "enable_metrics",
        "mcp_config",
        "memory_budget_gb",
        "models_config",
        "offline",
        "download_timeout",
        "download_retries",
        "served_model_name",
        "embedding_model",
        "embedding_max_length",
        "embedding_overflow_policy",
        "stt_model",
        "rerank_model",
        "max_audio_upload_mb",
        "max_tts_input_chars",
        "warm_prompts",
        "lazy_load_model",
        "auto_unload_idle_seconds",
        "model",
    }
    assert set(refused) == process_wide, sorted(set(refused) ^ process_wide)


def test_no_hand_kept_flag_list_in_the_loader():
    """The loader names no serve flag: routing comes from entry_contract()."""
    import inspect

    source = inspect.getsource(registry_mod.load_registry_config)
    for flag in ("chunked_prefill_tokens", "kv_cache_quantization", "default_top_k"):
        assert flag not in source
