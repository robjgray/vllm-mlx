# SPDX-License-Identifier: Apache-2.0
"""
Registry-backed multi-model serving with memory-budget eviction.

The registry maps OpenAI-compatible ``model`` names to concrete local paths or
declared HuggingFace IDs. Models are loaded lazily, optionally preloaded, and
evicted according to a memory-budget policy with configurable wait/fail/preempt
behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import difflib
import enum
import logging
import math
import os
import time
import types
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import TypeAdapter

from .api.models import ChatCompletionRequest
from .api.utils import is_mllm_model
from .cli_arg_types import parse_positive_finite_float
from .engine.base import BaseEngine, suspend_cancellation
from .lifecycle import ModelSpec
from .scheduler import SchedulerConfig
from .utils.download import DownloadConfig, ensure_model_downloaded

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


class ModelOwnershipError(RuntimeError):
    """Raised when an EngineCore attempts to use a model already in use."""


class _ModelOwnershipRegistry:
    """Process-local model ownership guard used by EngineCore."""

    def __init__(self) -> None:
        self._owners: dict[int, str] = {}

    def acquire(
        self,
        *,
        model: Any,
        engine: Any,
        engine_id: str,
        force: bool = True,
    ) -> None:
        key = id(model)
        owner = self._owners.get(key)
        if owner is not None and owner != engine_id and not force:
            raise ModelOwnershipError(
                f"Model is already owned by engine {owner}; "
                f"engine {engine_id} cannot acquire it"
            )
        self._owners[key] = engine_id

    def release(self, model: Any, engine_id: str) -> None:
        key = id(model)
        owner = self._owners.get(key)
        if owner == engine_id:
            self._owners.pop(key, None)

    def is_owned(self, model: Any) -> tuple[bool, str | None]:
        key = id(model)
        owner = self._owners.get(key)
        if owner is not None:
            return (True, owner)
        return (False, None)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._owners),
            "active_owners": len(self._owners),
        }


_ownership_registry = _ModelOwnershipRegistry()


def get_registry() -> _ModelOwnershipRegistry:
    """Return the global model ownership registry used by EngineCore."""
    return _ownership_registry


# ============================================================================
# Registry-backed multi-model serving types
# ============================================================================

ContentionStrategy = Literal[
    "fail",
    "wait",
    "preempt",
    "wait_then_fail",
    "wait_then_preempt",
]

EngineFactory = Callable[["ResolvedModelConfig"], BaseEngine]


# ============================================================================
# Entry contract: three typed sinks, each validating itself
# ============================================================================
#
# A registry entry is not a schema of its own. It is a set of keys routed to the
# three objects that already define what a *serve* invocation means:
#
#   scheduler  -- SchedulerConfig fields, applied with dataclasses.replace onto
#                 the entry's own clone (replace re-runs __post_init__, so the
#                 scheduler validates itself and unknown kwargs raise).
#   spec       -- ModelSpec fields, the engine-construction inputs the
#                 single-model path already uses (_build_engine).
#   request    -- request-time serve defaults, i.e. the values `serve <model>`
#                 puts in server.py module globals.
#
# Every sink derives its own key set from the object it feeds, so this module
# keeps no flag list. A key that reaches no sink is either a typo or a
# genuinely process-wide serve flag; both are refused loudly with different
# wording (see load_registry_config).

# YAML key -> the RegisteredModel field it means. Only keys whose spelling
# differs need an entry; negated twins map onto the positive field and invert.
ENTRY_ALIASES: dict[str, str] = {
    "mllm": "force_mllm",
    "specprefill": "specprefill_enabled",
}
_ENTRY_NEGATED: dict[str, str] = {
    "disable_prefix_cache": "enable_prefix_cache",
    "no_memory_aware_cache": "use_memory_aware_cache",
}

# How a ResolvedModelConfig becomes a ModelSpec. This mapping IS the list of
# spec fields the manager fills itself, so entry_contract() can exclude them
# from the spec sink without a second hand-kept list.
_SPEC_FROM_CONFIG: dict[str, str] = {
    "model_name": "resolved_source",
    "use_batching": "continuous_batching",
    "scheduler_config": "scheduler_config",
    "stream_interval": "stream_interval",
    "force_mllm": "force_mllm",
    "mtp": "enable_mtp",
    "prefill_step_size": "prefill_step_size",
    "specprefill_enabled": "specprefill_enabled",
    "specprefill_threshold": "specprefill_threshold",
    "specprefill_keep_pct": "specprefill_keep_pct",
    "specprefill_backbone_pct": "specprefill_backbone_pct",
    "specprefill_draft_model": "specprefill_draft_model",
    "prefix_trie_cache": "prefix_trie_cache",
    "prefix_trie_cache_size": "prefix_trie_cache_size",
    "prefix_trie_cache_memory_mb": "prefix_trie_cache_memory_mb",
    "gpu_memory_utilization": "gpu_memory_utilization",
    "mllm_draft_model": "mllm_draft_model",
    "mllm_draft_kind": "mllm_draft_kind",
    "mllm_draft_block_size": "mllm_draft_block_size",
    "default_mllm_draft": "default_mllm_draft",
    "reasoning_parser": "reasoning_parser",
    "tool_call_parser": "tool_call_parser",
    "enable_auto_tool_choice": "enable_auto_tool_choice",
}
# Spec fields that are identity/bookkeeping, never an entry's business.
_SPEC_IDENTITY = frozenset({"model_key", "model_path"})

# Keys that name the model rather than configure it.
ENTRY_META_KEYS = frozenset(
    {"name", "path", "source", "model", "preload", "estimated_memory_gb"}
)



@dataclass(frozen=True)
class RegistryRequestDefaults:
    """One entry's request-time defaults, mirroring serve <model>'s globals."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    max_request_tokens: int | None = None
    chat_template_kwargs: dict[str, object] | None = None
    thinking_token_budget: int | None = None
    timeout: float | None = None


def _serve_dests() -> dict[str, bool]:
    """Serve subparser dest -> True when it is a store_true switch.

    Imported lazily: cli imports this module from inside serve_command().
    """
    from .cli import create_parser

    parser = create_parser()
    for action in parser._subparsers._group_actions:  # noqa: SLF001 - argparse
        if getattr(action, "choices", None) and "serve" in action.choices:
            serve = action.choices["serve"]
            return {
                a.dest: isinstance(a, argparse._StoreTrueAction)
                for a in serve._actions
                if a.dest != "help"
            }
    raise RuntimeError("serve subparser not found")


def _engine_params() -> set[str]:
    """Keyword names both engines accept, i.e. what _build_engine may pass."""
    import inspect

    from .engine.batched import BatchedEngine
    from .engine.simple import SimpleEngine

    def names(cls: type) -> set[str]:
        return {
            name
            for name, param in inspect.signature(cls.__init__).parameters.items()
            if param.kind
            in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            and name != "self"
        }

    return names(SimpleEngine) | names(BatchedEngine)


def entry_contract() -> dict[str, dict[str, Any]]:
    """The three sinks' live key sets, derived from the objects themselves.

    ``spec`` is ModelSpec fields the engines actually take (what _build_engine
    forwards), so a ModelSpec field that is bookkeeping (``max_tokens`` is
    read from the server globals, not from the spec) cannot masquerade as an
    engine input. ``request`` is the serve dests that name a request-time
    default, crossed with the field the resolvers read.
    """
    scheduler = {f.name: f.type for f in dataclasses.fields(SchedulerConfig)}
    engine_params = _engine_params()
    derived = set(_SPEC_FROM_CONFIG) | _SPEC_IDENTITY
    spec = {
        f.name: f.type
        for f in dataclasses.fields(ModelSpec)
        if f.name in engine_params and f.name not in derived
    }
    typed = {f.name: f.type for f in dataclasses.fields(RegisteredModel)}
    defaults = RegistryRequestDefaults.__annotations__
    request_fields = ChatCompletionRequest.model_fields
    request: dict[str, Any] = {}
    # max_request_tokens is the one request-time default with no request-model
    # field: a client sends max_tokens, which is checked against this ceiling.
    request["max_request_tokens"] = defaults["max_request_tokens"]
    for dest in _serve_dests():
        if dest in typed or dest in ENTRY_ALIASES or dest in _ENTRY_NEGATED:
            continue
        short = dest.removeprefix("default_")
        if short in defaults and short in request_fields:
            request[dest] = defaults[short]
    return {
        "typed": typed,
        "scheduler": scheduler,
        "spec": spec,
        "request": request,
    }


def _coerce(entry_name: str, key: str, value: object, hint: Any) -> Any:
    """Validate one value against the annotation of the field it feeds."""
    try:
        return TypeAdapter(hint).validate_python(value)
    except Exception as exc:  # pydantic raises several subclasses
        raise ValueError(
            f"models-config entry '{entry_name}': {key}={value!r} is not valid "
            f"for type {hint}: {exc}"
        ) from None


def _sinks_for(
    key: str, contract: dict[str, dict[str, Any]], typed_names: set[str]
) -> list[tuple[str, str, Any]]:
    """Route one entry key to every sink that consumes it.

    Several keys feed two sinks at once: prefill_step_size is a SchedulerConfig
    field and an engine argument, mllm_draft_model is an engine argument and a
    ModelSpec field, enable_mtp is a SchedulerConfig field and SimpleEngine's
    mtp. Routing to one would leave the other silently at its default.
    """
    name = ENTRY_ALIASES.get(key, key)
    if name in typed_names:
        # A typed field (including an aliased one: mllm -> force_mllm) carries
        # its own validation in load_registry_config.
        return []
    routed: list[tuple[str, str, Any]] = []
    if key in _ENTRY_NEGATED:
        positive = _ENTRY_NEGATED[key]
        if positive in contract["scheduler"]:
            routed.append(("scheduler", positive, contract["scheduler"][positive]))
        return routed
    if name in contract["scheduler"]:
        routed.append(("scheduler", name, contract["scheduler"][name]))
    if name in contract["spec"]:
        routed.append(("spec", name, contract["spec"][name]))
    if key in contract["request"]:
        routed.append(
            ("request", key.removeprefix("default_"), contract["request"][key])
        )
    return routed


@dataclass(frozen=True)
class RegistryServeDefaults:
    """Global serve defaults inherited by registry entries."""

    continuous_batching: bool
    force_mllm: bool
    enable_mtp: bool
    prefill_step_size: int
    specprefill_enabled: bool
    specprefill_threshold: int
    specprefill_keep_pct: float
    specprefill_backbone_pct: float
    specprefill_draft_model: str | None
    prefix_trie_cache: bool
    prefix_trie_cache_size: int
    prefix_trie_cache_memory_mb: int | None
    stream_interval: int
    gpu_memory_utilization: float
    scheduler_config: SchedulerConfig | None
    max_tokens: int
    download_config: DownloadConfig
    auto_unload_idle_seconds: float = 0.0
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    enable_auto_tool_choice: bool = False

@dataclass(frozen=True)
class ContentionPolicy:
    """Policy used when a new model cannot fit inside the memory budget."""

    strategy: ContentionStrategy = "wait_then_fail"
    wait_timeout_s: float | None = 30.0
    preempt_after_s: float | None = None


@dataclass(frozen=True)
class RegistryManagerConfig:
    """Global registry manager configuration."""

    memory_budget_bytes: int
    policy: ContentionPolicy
    idle_unload_seconds: float = 0.0


@dataclass(frozen=True)
class RegisteredModel:
    """One configured model entry."""

    name: str
    source: str
    preload: bool = False
    continuous_batching: bool | None = None
    force_mllm: bool | None = None
    enable_mtp: bool | None = None
    prefill_step_size: int | None = None
    specprefill_enabled: bool | None = None
    specprefill_threshold: int | None = None
    specprefill_keep_pct: float | None = None
    specprefill_backbone_pct: float | None = None
    specprefill_draft_model: str | None = None
    mllm_draft_model: str | None = None
    mllm_draft_kind: str | None = None
    mllm_draft_block_size: int | None = None
    default_mllm_draft: bool | None = None
    prefix_trie_cache: bool | None = None
    prefix_trie_cache_size: int | None = None
    prefix_trie_cache_memory_mb: int | None = None
    stream_interval: int | None = None
    gpu_memory_utilization: float | None = None
    estimated_memory_bytes: int | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    enable_auto_tool_choice: bool | None = None
    cache_memory_mb: int | None = None
    # SchedulerConfig fields as {field: value}; applied to this entry's own
    # clone with dataclasses.replace, so the scheduler validates them.
    scheduler: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Request-time defaults (server.py's per-request globals).
    request_defaults: "RegistryRequestDefaults | None" = None
    # ModelSpec fields with no dedicated field above, passed to _build_engine.
    spec_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclass(frozen=True)
class ResolvedModelConfig:
    """Effective configuration for a loaded model."""

    entry: RegisteredModel
    resolved_source: str
    continuous_batching: bool
    force_mllm: bool
    enable_mtp: bool
    prefill_step_size: int
    specprefill_enabled: bool
    specprefill_threshold: int
    specprefill_keep_pct: float
    specprefill_backbone_pct: float
    specprefill_draft_model: str | None
    mllm_draft_model: str | None
    mllm_draft_kind: str | None
    mllm_draft_block_size: int | None
    default_mllm_draft: bool
    prefix_trie_cache: bool
    prefix_trie_cache_size: int
    prefix_trie_cache_memory_mb: int | None
    stream_interval: int
    gpu_memory_utilization: float
    scheduler_config: SchedulerConfig | None
    estimated_memory_bytes: int
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    enable_auto_tool_choice: bool = False
    cache_memory_mb: int | None = None

@dataclass
class LoadedModel:
    """Runtime state for a loaded engine."""

    config: ResolvedModelConfig
    engine: BaseEngine
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    active_requests: int = 0
    active_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    preempting: bool = False


@dataclass
class PendingLoad:
    """A reserved model load in progress."""

    model_name: str
    required_bytes: int
    future: asyncio.Future[LoadedModel]


@dataclass
class ModelLease:
    """Active lease for a loaded model."""

    manager: "ModelManager | None"
    model_name: str
    engine: BaseEngine
    release_cb: Callable[[], Awaitable[None]]

    async def release(self) -> None:
        if self.manager is None:
            return
        manager = self.manager
        self.manager = None
        await self.release_cb()

    async def __aenter__(self) -> "ModelLease":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()


def _clone_scheduler_config(
    config: SchedulerConfig | None, *, prefill_step_size: int
) -> SchedulerConfig | None:
    """Clone a SchedulerConfig so per-model overrides do not mutate globals."""
    if config is None:
        return None
    values = vars(config).copy()
    values["prefill_step_size"] = prefill_step_size
    return SchedulerConfig(**values)


def _parse_memory_budget_bytes(value: Any) -> int:
    """Parse a memory budget from bytes, MB, or GB."""
    if value is None:
        raise ValueError("models-config manager.memory_budget_gb is required")
    multiplier = 1024**3
    amount: str | int | float
    if isinstance(value, (int, float)):
        amount = value
    elif isinstance(value, str):
        raw = value.strip().lower()
        if raw.endswith("gb"):
            amount = raw[:-2]
        elif raw.endswith("mb"):
            amount = raw[:-2]
            multiplier = 1024**2
        elif raw.endswith("b"):
            amount = raw[:-1]
            multiplier = 1
        else:
            amount = raw
    else:
        raise TypeError(f"Unsupported memory budget value: {value!r}")

    value_name = "models-config manager memory budget"
    parsed = parse_positive_finite_float(amount, value_name)
    bytes_value = parse_positive_finite_float(parsed * multiplier, value_name)
    memory_budget_bytes = int(bytes_value)
    if memory_budget_bytes == 0:
        raise ValueError(f"{value_name} must be at least 1 byte")
    return memory_budget_bytes


def _safe_available_memory_bytes() -> int:
    """Best-effort available system memory."""
    if psutil is None:  # pragma: no cover - fallback only
        return 0
    return int(psutil.virtual_memory().available)


def _device_working_set_bytes() -> int | None:
    """Best-effort Metal recommended working-set size, or None when unavailable."""
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return None
        info = mx.device_info()
        raw = info.get(
            "max_recommended_working_set_size",
            info.get("memory_size", 0),
        )
        working_set = int(raw or 0)
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.debug("Could not query MLX device memory: %s", exc)
        return None
    return working_set or None


@dataclass(frozen=True)
class MemoryBudgetReport:
    """Reconciliation of the manager weight budget with the Metal ceiling.

    The manager budget counts model *weights* only, and the Metal allocation
    ceiling (``gpu_memory_utilization`` x device working set) is process-wide.
    Those two are directly comparable, so a budget above the ceiling is a
    deterministic conflict: the manager will keep models resident that MLX
    cannot allocate, and the load fails instead of evicting.

    The prefix-cache limit is deliberately *not* folded into that comparison.
    ``cache_memory_mb`` is a per-engine maximum — it is cloned into each
    applicable resident continuous-batching engine and allocated lazily, and
    simple-mode entries never receive it at all — so it is neither a single
    process-wide reservation nor a bound that can be subtracted once. It is
    reported alongside the ceiling instead, with its own conflict check.
    """

    budget_bytes: int
    device_working_set_bytes: int | None
    gpu_memory_utilization: float | None
    gpu_memory_utilization_source: str | None
    per_engine_cache_limit_bytes: int | None
    per_engine_cache_percent: float | None
    memory_aware_prefix_cache_entries: int
    continuous_batching_entries: int
    total_entries: int

    @property
    def allocation_ceiling_bytes(self) -> int | None:
        """Metal soft allocation limit that will be installed at engine start.

        ``None`` when no ceiling can be attributed: either MLX cannot report a
        device working set, or no entry will install one (only ``BatchedEngine``
        calls ``mx.set_memory_limit``).
        """
        if self.device_working_set_bytes is None or self.gpu_memory_utilization is None:
            return None
        return int(self.device_working_set_bytes * self.gpu_memory_utilization)

    @property
    def exceeds_ceiling(self) -> bool:
        """True when the weights budget alone cannot fit under the ceiling.

        Both sides are process-wide totals, so this is the deterministic check.
        """
        ceiling = self.allocation_ceiling_bytes
        return ceiling is not None and self.budget_bytes > ceiling

    @property
    def cache_limit_exceeds_ceiling(self) -> bool:
        """True when one engine's prefix cache could alone fill the ceiling."""
        ceiling = self.allocation_ceiling_bytes
        if ceiling is None or self.per_engine_cache_limit_bytes is None:
            return False
        return self.per_engine_cache_limit_bytes >= ceiling


def _registry_entry_uses_mllm_path(
    entry: RegisteredModel, defaults: RegistryServeDefaults
) -> bool:
    """Classify explicit or locally verifiable MLLM registry entries."""
    force_mllm = (
        entry.force_mllm if entry.force_mllm is not None else defaults.force_mllm
    )
    if force_mllm:
        return True

    # Remote IDs are unresolved when this startup report is built. Their names
    # are only a heuristic and can disagree with the downloaded config that the
    # engine later treats as authoritative. Require an explicit declaration for
    # remote entries so capacity warnings cannot be false positives.
    if not Path(entry.source).exists():
        return False

    return is_mllm_model(entry.source)


def build_memory_budget_report(
    manager_config: RegistryManagerConfig,
    registry: dict[str, RegisteredModel],
    defaults: RegistryServeDefaults,
    *,
    device_working_set_bytes: int | None = None,
) -> MemoryBudgetReport:
    """Reconcile the manager weight budget against the Metal allocation ceiling.

    The Metal limit is process-wide but is re-installed by every
    ``BatchedEngine`` start, so the ceiling the manager has to live under is the
    *lowest* utilization among the entries that actually install one. Only
    continuous-batching entries qualify: ``SimpleEngine`` never calls
    ``mx.set_memory_limit`` and is not even given a ``gpu_memory_utilization``.
    A registry with no continuous-batching entries therefore gets no attributed
    ceiling rather than one derived from a value nothing installs.
    """
    if device_working_set_bytes is None:
        device_working_set_bytes = _device_working_set_bytes()

    # Only BatchedEngine installs a Metal allocation limit — SimpleEngine is not
    # even constructed with a gpu_memory_utilization — so a simple-mode entry's
    # override is inert and must not be treated as a ceiling candidate.
    candidates: list[tuple[float, int, str]] = []
    memory_aware_prefix_cache_entries = 0
    scheduler_config = defaults.scheduler_config
    if scheduler_config is None:
        # A registry entry can enable continuous batching while the global CLI
        # flag is off. BatchedEngine supplies SchedulerConfig defaults in that
        # case, so the startup report must account for the same default cache.
        scheduler_config = SchedulerConfig()
    for name in sorted(registry):
        entry = registry[name]
        entry_continuous_batching = (
            entry.continuous_batching
            if entry.continuous_batching is not None
            else defaults.continuous_batching
        )
        if not entry_continuous_batching:
            continue

        if (
            scheduler_config is not None
            and getattr(scheduler_config, "enable_prefix_cache", False)
            and getattr(scheduler_config, "use_memory_aware_cache", False)
        ):
            uses_mllm_path = _registry_entry_uses_mllm_path(entry, defaults)
            if uses_mllm_path or not getattr(
                scheduler_config, "use_paged_cache", False
            ):
                memory_aware_prefix_cache_entries += 1

        if entry.gpu_memory_utilization is not None:
            # Rank 1: an override is only attributable to the entry declaring it.
            candidates.append(
                (entry.gpu_memory_utilization, 1, f"models-config entry '{name}'")
            )
        else:
            # Rank 0: prefer the serve default as the named source on ties.
            candidates.append((defaults.gpu_memory_utilization, 0, "serve default"))

    continuous_batching_entries = len(candidates)

    utilization: float | None = None
    utilization_source: str | None = None
    if candidates:
        utilization, _, utilization_source = min(candidates)

    # cache_memory_mb only binds for continuous-batching engines that actually
    # construct the memory-aware prefix cache. Text engines select the paged
    # cache when requested, but MLLM engines do not implement that cache and
    # continue to construct MemoryAwarePrefixCache.
    per_engine_cache_limit_bytes: int | None = None
    per_engine_cache_percent: float | None = None
    cache_applies = (
        scheduler_config is not None and memory_aware_prefix_cache_entries > 0
    )
    if cache_applies:
        # Per-entry overrides install a different limit per engine; the report
        # line is informational and the warning compares one limit against the
        # process-wide ceiling, so report the largest limit any entry will
        # install. Entries without an override use the global scheduler value.
        per_entry_cache_mb: list[int] = []
        for name in sorted(registry):
            entry = registry[name]
            entry_continuous_batching = (
                entry.continuous_batching
                if entry.continuous_batching is not None
                else defaults.continuous_batching
            )
            if not entry_continuous_batching:
                continue
            if (
                scheduler_config is not None
                and getattr(scheduler_config, "enable_prefix_cache", False)
                and getattr(scheduler_config, "use_memory_aware_cache", False)
                and (
                    _registry_entry_uses_mllm_path(entry, defaults)
                    or not getattr(scheduler_config, "use_paged_cache", False)
                )
            ):
                per_entry_cache_mb.append(
                    entry.cache_memory_mb
                    if entry.cache_memory_mb is not None
                    else getattr(scheduler_config, "cache_memory_mb", None)
                )
        cache_memory_mb = max(
            (value for value in per_entry_cache_mb if value), default=None
        )
        if cache_memory_mb:
            per_engine_cache_limit_bytes = int(cache_memory_mb) * (1024**2)
        else:
            percent = getattr(scheduler_config, "cache_memory_percent", None)
            if percent:
                per_engine_cache_percent = float(percent)

    return MemoryBudgetReport(
        budget_bytes=manager_config.memory_budget_bytes,
        device_working_set_bytes=device_working_set_bytes,
        gpu_memory_utilization=utilization,
        gpu_memory_utilization_source=utilization_source,
        per_engine_cache_limit_bytes=per_engine_cache_limit_bytes,
        per_engine_cache_percent=per_engine_cache_percent,
        memory_aware_prefix_cache_entries=memory_aware_prefix_cache_entries,
        continuous_batching_entries=continuous_batching_entries,
        total_entries=len(registry),
    )


def log_memory_budget_report(report: MemoryBudgetReport) -> None:
    """Log the budget/ceiling reconciliation, warning when they conflict."""
    gb = 1024**3
    ceiling = report.allocation_ceiling_bytes

    if ceiling is None:
        if report.gpu_memory_utilization is None:
            reason = (
                "no continuous-batching entries, and --gpu-memory-utilization "
                "installs a Metal limit only for those"
            )
        else:
            reason = "MLX cannot report a device working set on this host"
        logger.info(
            "Registry memory budget: %.1f GB of model weights; no Metal "
            "allocation ceiling to reconcile it with (%s)",
            report.budget_bytes / gb,
            reason,
        )
        return

    engines = (
        f"{report.memory_aware_prefix_cache_entries} of "
        f"{report.total_entries} entries"
    )
    if report.per_engine_cache_limit_bytes is not None:
        cache_desc = (
            f"{report.per_engine_cache_limit_bytes / gb:.1f} GB per "
            f"memory-aware prefix-cache engine (--cache-memory-mb, {engines})"
        )
    elif report.per_engine_cache_percent is not None:
        cache_desc = (
            f"~{report.per_engine_cache_percent * 100:.0f}% of available RAM per "
            f"memory-aware prefix-cache engine (--cache-memory-percent, {engines}); "
            "scales at runtime"
        )
    else:
        cache_desc = "none configured"

    logger.info(
        "Registry memory budget: %.1f GB of model weights; "
        "Metal allocation ceiling %.1f GB (%.0f%% of %.1f GB, from %s); "
        "prefix-cache maximum %s",
        report.budget_bytes / gb,
        ceiling / gb,
        report.gpu_memory_utilization * 100,
        (report.device_working_set_bytes or 0) / gb,
        report.gpu_memory_utilization_source,
        cache_desc,
    )

    if report.exceeds_ceiling:
        logger.warning(
            "models-config manager.memory_budget_gb (%.1f GB) exceeds the Metal "
            "allocation ceiling (%.1f GB). The budget counts model weights only, "
            "so the manager will keep models resident that MLX cannot allocate, "
            "and a load can fail with an out-of-memory error instead of evicting. "
            "Lower the budget below %.1f GB — further still, since the KV cache "
            "and activations also come out of the ceiling — or raise "
            "--gpu-memory-utilization.",
            report.budget_bytes / gb,
            ceiling / gb,
            ceiling / gb,
        )

    if report.cache_limit_exceeds_ceiling:
        logger.warning(
            "--cache-memory-mb (%.1f GB per memory-aware prefix-cache engine) is "
            "at or "
            "above the Metal allocation ceiling (%.1f GB) on its own, leaving no "
            "room for model weights. Note this is a per-engine maximum: it is "
            "cloned into every applicable resident continuous-batching engine, so the "
            "aggregate grows with the number of resident models.",
            (report.per_engine_cache_limit_bytes or 0) / gb,
            ceiling / gb,
        )

    if not report.exceeds_ceiling and not report.cache_limit_exceeds_ceiling:
        logger.info(
            "The registry budget covers model weights only; the KV cache, the "
            "prefix cache and activations are additional and are not reserved "
            "by it."
        )


def _estimate_model_bytes_from_source(source: str) -> int:
    """Estimate model footprint from local artifact size when possible."""
    path = Path(source)
    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size if path.suffix in {".safetensors", ".gguf"} else 0

    total = 0
    for pattern in ("*.safetensors", "*.gguf"):
        for fp in path.rglob(pattern):
            try:
                total += fp.stat().st_size
            except OSError:
                continue
    return total


def load_registry_config(
    config_path: str | os.PathLike[str],
    defaults: RegistryServeDefaults,
    *,
    memory_budget_gb: float | None = None,
) -> tuple[RegistryManagerConfig, dict[str, RegisteredModel]]:
    """Load and validate the models registry YAML file."""
    import yaml  # lazy: only needed when a registry config is provided

    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models-config must define a non-empty 'models' list")

    manager_raw = raw.get("manager") or {}
    policy_raw = manager_raw.get("contention_policy") or {}
    policy = ContentionPolicy(
        strategy=policy_raw.get("strategy", "wait_then_fail"),
        wait_timeout_s=(
            float(policy_raw["wait_timeout_s"])
            if policy_raw.get("wait_timeout_s") is not None
            else 30.0
        ),
        preempt_after_s=(
            float(policy_raw["preempt_after_s"])
            if policy_raw.get("preempt_after_s") is not None
            else None
        ),
    )
    if policy.strategy not in {
        "fail",
        "wait",
        "preempt",
        "wait_then_fail",
        "wait_then_preempt",
    }:
        raise ValueError(f"Unsupported contention strategy: {policy.strategy}")

    idle_unload_seconds = manager_raw.get("idle_unload_seconds")
    manager = RegistryManagerConfig(
        memory_budget_bytes=_parse_memory_budget_bytes(
            memory_budget_gb
            if memory_budget_gb is not None
            else manager_raw.get("memory_budget_gb", manager_raw.get("memory_budget"))
        ),
        policy=policy,
        idle_unload_seconds=(
            float(idle_unload_seconds)
            if idle_unload_seconds is not None
            else defaults.auto_unload_idle_seconds
        ),
    )

    registry: dict[str, RegisteredModel] = {}
    # Derived once, not per entry: the routing contract and the serve dests.
    contract = entry_contract()
    typed_names = set(contract["typed"])
    serve_dests = _serve_dests()
    for item in models:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid model entry: {item!r}")
        name = item.get("name")
        source = item.get("path") or item.get("source") or item.get("model")
        if not name or not source:
            raise ValueError(
                f"Each model entry must define 'name' and one of 'path'/'source'/'model': {item!r}"
            )
        if name in registry:
            raise ValueError(f"Duplicate model name in registry: {name}")

        estimated = item.get("estimated_memory_gb")
        estimated_bytes = (
            int(float(estimated) * (1024**3)) if estimated is not None else None
        )

        raw_gpu_memory_utilization = item.get("gpu_memory_utilization")
        gpu_memory_utilization = None
        if raw_gpu_memory_utilization is not None:
            try:
                gpu_memory_utilization = float(raw_gpu_memory_utilization)
            except (TypeError, ValueError):
                raise ValueError(
                    f"models-config entry '{name}' gpu_memory_utilization must "
                    "be finite and within (0, 1]"
                ) from None
            if not math.isfinite(gpu_memory_utilization) or not (
                0.0 < gpu_memory_utilization <= 1.0
            ):
                raise ValueError(
                    f"models-config entry '{name}' gpu_memory_utilization must "
                    "be finite and within (0, 1]"
                )

        raw_mllm_draft_block_size = item.get("mllm_draft_block_size")
        mllm_draft_block_size = None
        if raw_mllm_draft_block_size is not None:
            try:
                mllm_draft_block_size = int(raw_mllm_draft_block_size)
            except (TypeError, ValueError):
                raise ValueError(
                    f"models-config entry '{name}': "
                    "mllm_draft_block_size must be a positive integer"
                ) from None
            if mllm_draft_block_size <= 0:
                raise ValueError(
                    f"models-config entry '{name}': "
                    "mllm_draft_block_size must be a positive integer"
                )

        mllm_draft_model = item.get("mllm_draft_model")
        if mllm_draft_model is not None:
            mllm_draft_model = str(mllm_draft_model)
        mllm_draft_kind = item.get("mllm_draft_kind")
        if mllm_draft_kind is not None:
            mllm_draft_kind = str(mllm_draft_kind)
        default_mllm_draft = item.get("default_mllm_draft")

        reasoning_parser = item.get("reasoning_parser")
        if reasoning_parser is not None and (
            not isinstance(reasoning_parser, str) or not reasoning_parser.strip()
        ):
            raise ValueError(
                f"models-config entry '{name}': "
                "reasoning_parser must be a non-empty string"
            )
        tool_call_parser = item.get("tool_call_parser")
        if tool_call_parser is not None and (
            not isinstance(tool_call_parser, str) or not tool_call_parser.strip()
        ):
            raise ValueError(
                f"models-config entry '{name}': "
                "tool_call_parser must be a non-empty string"
            )
        enable_auto_tool_choice = item.get("enable_auto_tool_choice")
        if enable_auto_tool_choice is not None and not isinstance(
            enable_auto_tool_choice, bool
        ):
            raise ValueError(
                f"models-config entry '{name}': "
                "enable_auto_tool_choice must be a boolean"
            )
        if enable_auto_tool_choice and not tool_call_parser:
            raise ValueError(
                f"models-config entry '{name}': "
                "enable_auto_tool_choice requires tool_call_parser"
            )
        raw_cache_memory_mb = item.get("cache_memory_mb")
        cache_memory_mb = None
        if raw_cache_memory_mb is not None:
            try:
                cache_memory_mb = int(raw_cache_memory_mb)
            except (TypeError, ValueError):
                raise ValueError(
                    f"models-config entry '{name}': "
                    "cache_memory_mb must be a positive integer"
                ) from None
            if cache_memory_mb <= 0:
                raise ValueError(
                    f"models-config entry '{name}': "
                    "cache_memory_mb must be a positive integer"
                )

        if mllm_draft_model and not item.get("mllm"):
            raise ValueError(
                f"models-config entry '{name}': mllm_draft_model requires mllm: true"
            )
        if default_mllm_draft and not mllm_draft_model:
            raise ValueError(
                f"models-config entry '{name}': "
                "default_mllm_draft requires mllm_draft_model"
            )
        resolved_continuous_batching = (
            item.get("continuous_batching")
            if item.get("continuous_batching") is not None
            else defaults.continuous_batching
        )
        if (
            mllm_draft_model
            and resolved_continuous_batching
            and mllm_draft_kind != "mtp"
        ):
            raise ValueError(
                f"models-config entry '{name}': mllm_draft_model with "
                "continuous_batching requires mllm_draft_kind: mtp"
            )
        if mllm_draft_model and manager.idle_unload_seconds > 0:
            raise ValueError(
                f"models-config entry '{name}': mllm_draft_model is not supported "
                "with idle_unload_seconds > 0 (lifecycle residency does not "
                "support MLLM draft models yet)"
            )

        scheduler_bag: dict[str, Any] = {}
        spec_bag: dict[str, Any] = {}
        request_bag: dict[str, Any] = {}
        for key, value in item.items():
            if key in ENTRY_META_KEYS or ENTRY_ALIASES.get(key, key) in typed_names:
                continue
            routed = _sinks_for(key, contract, typed_names)
            raw = not bool(value) if key in _ENTRY_NEGATED else value
            for sink, field_name, hint in routed:
                coerced = _coerce(name, key, raw, hint)
                if sink == "scheduler":
                    # Record every value the entry wrote, including False: the
                    # loop only visits keys the entry actually set, so present-
                    # and-false is distinguishable from absent, and dropping a
                    # false would make disable_prefix_cache: true a silent no-op.
                    scheduler_bag[field_name] = coerced
                elif sink == "spec":
                    spec_bag[field_name] = coerced
                else:
                    request_bag[field_name] = coerced

        # Anything left over is either a typo or a serve flag with one value
        # for the whole process. Both are refused; the wording distinguishes
        # them and the suggestion list comes from the sinks, not a flag table.
        known = (
            typed_names
            | set(ENTRY_META_KEYS)
            | set(ENTRY_ALIASES)
            | set(_ENTRY_NEGATED)
            | set(contract["scheduler"])
            | set(contract["spec"])
            | set(contract["request"])
        )
        unknown = sorted(
            key
            for key in item
            if key not in known and key not in {"estimated_memory_bytes", "manager",
                                                "estimated_memory_gb"}
        )
        process_wide = [key for key in unknown if key in serve_dests]
        typos = [key for key in unknown if key not in serve_dests]
        if typos:
            suggestions = {
                key: difflib.get_close_matches(key, sorted(known), n=1)
                for key in typos
            }
            hints = " ".join(
                f"Did you mean '{match[0]}' instead of '{key}'?"
                for key, match in suggestions.items()
                if match
            )
            raise ValueError(
                f"models-config entry '{name}': unknown key(s) {typos}. {hints}".strip()
            )
        if process_wide:
            raise ValueError(
                f"models-config entry '{name}': {process_wide} set the whole "
                "server (one host, one port, one auth surface, one download "
                "config), so they cannot be per-entry. Set them on the serve "
                "command line instead; an entry may only set SchedulerConfig "
                "fields, engine fields, and request defaults."
            )

        # Same invariant the single-model path enforces in load_model(): the
        # default output budget can never exceed the per-request ceiling.
        if (
            request_bag.get("max_tokens") is not None
            and request_bag.get("max_request_tokens") is not None
            and request_bag["max_tokens"] > request_bag["max_request_tokens"]
        ):
            raise ValueError(
                f"models-config entry {name!r}: max_tokens ({request_bag['max_tokens']}) "
                f"cannot exceed max_request_tokens ({request_bag['max_request_tokens']})"
            )

        registry[name] = RegisteredModel(
            name=name,
            source=str(source),
            scheduler=scheduler_bag,
            request_defaults=(
                RegistryRequestDefaults(**request_bag) if request_bag else None
            ),
            spec_overrides=spec_bag,
            preload=bool(item.get("preload", False)),
            continuous_batching=item.get("continuous_batching"),
            force_mllm=item.get("mllm"),
            enable_mtp=item.get("enable_mtp"),
            prefill_step_size=item.get("prefill_step_size"),
            specprefill_enabled=item.get("specprefill"),
            specprefill_threshold=item.get("specprefill_threshold"),
            specprefill_keep_pct=item.get("specprefill_keep_pct"),
            specprefill_backbone_pct=item.get("specprefill_backbone_pct"),
            specprefill_draft_model=item.get("specprefill_draft_model"),
            mllm_draft_model=mllm_draft_model,
            mllm_draft_kind=mllm_draft_kind,
            mllm_draft_block_size=mllm_draft_block_size,
            default_mllm_draft=default_mllm_draft,
            prefix_trie_cache=item.get("prefix_trie_cache"),
            prefix_trie_cache_size=item.get("prefix_trie_cache_size"),
            prefix_trie_cache_memory_mb=item.get("prefix_trie_cache_memory_mb"),
            stream_interval=item.get("stream_interval"),
            gpu_memory_utilization=gpu_memory_utilization,
            estimated_memory_bytes=estimated_bytes,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            enable_auto_tool_choice=enable_auto_tool_choice,
            cache_memory_mb=cache_memory_mb,
        )

    return manager, registry


class ModelManager:
    """Registry-backed model manager with lazy load and memory-budget eviction."""

    def __init__(
        self,
        manager_config: RegistryManagerConfig,
        registry: dict[str, RegisteredModel],
        defaults: RegistryServeDefaults,
        *,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self._config = manager_config
        self._registry = registry
        self._defaults = defaults
        self._engine_factory = engine_factory
        self._loaded: dict[str, LoadedModel] = {}
        self._loading: dict[str, PendingLoad] = {}
        self._unloading: dict[str, LoadedModel] = {}
        self._condition = asyncio.Condition()
        self._shutting_down = False

    @property
    def memory_budget_bytes(self) -> int:
        return self._config.memory_budget_bytes

    @property
    def idle_unload_seconds(self) -> float:
        return self._config.idle_unload_seconds

    @property
    def registered_model_names(self) -> list[str]:
        """Return sorted list of all registered model names."""
        return sorted(self._registry.keys())

    def has_model(self, model_name: str) -> bool:
        return model_name in self._registry

    def request_defaults(self, model_name: str) -> "RegistryRequestDefaults | None":
        """The entry's request-time defaults, or None (single-model serve)."""
        entry = self._registry.get(model_name)
        return entry.request_defaults if entry is not None else None

    def list_models(self) -> list[dict[str, Any]]:
        """Return registry state for /v1/models."""
        data = []
        for name, entry in self._registry.items():
            loaded = self._loaded.get(name)
            dead = loaded is not None and not loaded.engine.is_healthy()
            unloading = self._unloading.get(name)
            loading = self._loading.get(name)
            state = "unloaded"
            if dead:
                # Reported truthfully at read time; the stale entry itself is
                # reaped the next time acquire() is called for this model.
                state = "dead"
            elif loaded is not None:
                state = "preempting" if loaded.preempting else "loaded"
            elif loading is not None:
                state = "loading"
            elif unloading is not None:
                state = "unloading"

            estimated = (
                loaded.config.estimated_memory_bytes
                if loaded is not None
                else (
                    unloading.config.estimated_memory_bytes
                    if unloading is not None
                    else (
                        loading.required_bytes
                        if loading is not None
                        else self._resolve_estimated_bytes(entry, entry.source)
                    )
                )
            )
            data.append(
                {
                    "id": name,
                    "status": state,
                    "loaded": loaded is not None and not dead,
                    "owned_by": "vllm-mlx",
                    "source": entry.source,
                    "memory_gb": round(estimated / (1024**3), 2) if estimated else None,
                    "last_used_at": loaded.last_used_at if loaded is not None else None,
                }
            )
        return data

    def get_metrics_engine(self) -> BaseEngine | None:
        """Return the engine to source Prometheus gauges from, or None if idle.

        The gauges this feeds (``/metrics``) are unlabeled scalars with no
        ``model=`` dimension, so they can only ever describe one engine at a
        time. With a single resident model that engine is the obvious and
        exact choice. With more than one resident, there is no way to report
        all of them without a schema change, so this reports the most
        recently used one as a representative sample rather than reporting
        nothing.
        """
        if not self._loaded:
            return None
        return max(self._loaded.values(), key=lambda loaded: loaded.last_used_at).engine

    async def preload(self) -> None:
        """Preload any entries marked preload=true."""
        for entry in self._registry.values():
            if entry.preload:
                lease = await self.acquire(entry.name)
                await lease.release()

    async def shutdown(self) -> None:
        """Stop and unload every loaded engine."""
        pending: list[asyncio.Future[LoadedModel]] = []
        unloads: list[LoadedModel] = []
        cancel_tasks: set[asyncio.Task[Any]] = set()

        async with self._condition:
            self._shutting_down = True
            pending = [item.future for item in self._loading.values()]
            for loaded in self._loaded.values():
                loaded.preempting = True
                cancel_tasks.update(loaded.active_tasks)
            for name in list(self._loaded.keys()):
                if self._loaded[name].active_requests == 0:
                    unloads.append(self._begin_unload_locked(name))
            self._condition.notify_all()

        for task in cancel_tasks:
            task.cancel()
        await self._run_unloads(unloads)

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        remaining: list[LoadedModel] = []
        async with self._condition:
            for name in list(self._loaded.keys()):
                if self._loaded[name].active_requests == 0:
                    remaining.append(self._begin_unload_locked(name))
            self._condition.notify_all()

        await self._run_unloads(remaining)

    async def acquire(self, model_name: str) -> ModelLease:
        """Acquire a lease for a configured model."""
        if model_name not in self._registry:
            raise KeyError(model_name)

        start = time.monotonic()
        while True:
            load: PendingLoad | None = None
            unloads: list[LoadedModel] = []
            cancel_tasks: set[asyncio.Task[Any]] = set()
            same_model_future: asyncio.Future[LoadedModel] | None = None
            wait_timeout: float | None = None

            async with self._condition:
                if self._shutting_down:
                    raise RuntimeError("Model manager is shutting down")

                reaped = self._reap_dead_locked(model_name)
                if reaped is not None:
                    # Engine died without going through an explicit unload;
                    # tear it down like any other idle unload, then loop
                    # back around and load a fresh engine.
                    unloads = [reaped]
                else:
                    claimed = self._claim_loaded_locked(model_name)
                    if claimed is not None:
                        return claimed

                    same_model_future = self._loading.get(model_name, None)
                    if same_model_future is not None:
                        same_model_future = same_model_future.future
                    elif model_name in self._unloading:
                        wait_timeout = self._remaining_wait_timeout(start)
                    else:
                        entry = self._registry[model_name]
                        required_bytes = self._resolve_estimated_bytes(
                            entry, entry.source
                        )
                        unloads = self._collect_idle_unloads_locked(
                            model_name, required_bytes
                        )
                        if not unloads and self._can_reserve_locked(required_bytes):
                            load = self._reserve_load_locked(model_name, required_bytes)
                        elif not unloads:
                            cancel_tasks = self._maybe_preempt_locked(
                                model_name=model_name,
                                required_bytes=required_bytes,
                                start=start,
                            )
                            if not cancel_tasks and not self._should_wait_locked(start):
                                raise RuntimeError(
                                    f"Cannot load '{model_name}' within memory budget "
                                    f"({self._config.memory_budget_bytes / (1024**3):.1f} GB)"
                                )
                            wait_timeout = self._remaining_wait_timeout(start)

            if unloads:
                await self._run_unloads(unloads)
                continue

            if cancel_tasks:
                for task in cancel_tasks:
                    task.cancel()
                timeout = self._remaining_wait_timeout(start)
                await self._wait_for_change(timeout)
                continue

            if load is not None:
                loaded = await self._execute_load(load)
                async with self._condition:
                    claimed = self._claim_loaded_locked(
                        model_name, loaded_override=loaded
                    )
                    if claimed is not None:
                        return claimed
                continue

            if same_model_future is not None:
                loaded = await same_model_future
                async with self._condition:
                    claimed = self._claim_loaded_locked(
                        model_name, loaded_override=loaded
                    )
                    if claimed is not None:
                        return claimed
                continue

            await self._wait_for_change(wait_timeout)

    async def release(self, model_name: str) -> None:
        """Release a previously acquired model lease."""
        unload: LoadedModel | None = None

        async with self._condition:
            loaded = self._loaded.get(model_name)
            if loaded is None:
                return

            loaded.active_requests = max(0, loaded.active_requests - 1)
            loaded.last_used_at = time.time()
            task = asyncio.current_task()
            if task is not None:
                loaded.active_tasks.discard(task)

            if loaded.preempting and loaded.active_requests == 0:
                unload = self._begin_unload_locked(model_name)

            self._condition.notify_all()

        if unload is not None:
            await self._run_unloads([unload])

    async def unload_idle(self) -> list[str]:
        """Unload every loaded model idle past ``idle_unload_seconds``.

        No-op (returns an empty list) if idle-unload is disabled
        (``idle_unload_seconds <= 0``). Unlike memory-budget eviction, this
        proactively frees models even when no other model is being requested.
        Returns the names of models that were unloaded.
        """
        idle_seconds = self._config.idle_unload_seconds
        if idle_seconds <= 0:
            return []

        now = time.time()
        async with self._condition:
            stale = [
                loaded
                for loaded in self._idle_candidates_locked()
                if now - loaded.last_used_at >= idle_seconds
            ]
            unloads = [
                self._begin_unload_locked(loaded.config.entry.name) for loaded in stale
            ]
            self._condition.notify_all()

        if unloads:
            await self._run_unloads(unloads)

        return [loaded.config.entry.name for loaded in unloads]

    async def run_idle_reaper(self) -> None:
        """Background loop that proactively unloads idle models.

        Mirrors the single-model residency lifecycle loop: sleeps at half the
        configured idle timeout (bounded to 5s) so short timeouts stay
        responsive, and a failed pass is logged rather than killing the loop.
        """
        idle_seconds = self._config.idle_unload_seconds
        if idle_seconds <= 0:
            return

        while True:
            await asyncio.sleep(min(idle_seconds / 2, 5.0))
            try:
                await self.unload_idle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Idle unload pass failed")

    def _claim_loaded_locked(
        self,
        model_name: str,
        *,
        loaded_override: LoadedModel | None = None,
    ) -> ModelLease | None:
        loaded = loaded_override or self._loaded.get(model_name)
        if loaded is None:
            return None

        if loaded_override is not None and model_name not in self._loaded:
            self._loaded[model_name] = loaded

        if loaded.preempting:
            return None

        loaded.active_requests += 1
        loaded.last_used_at = time.time()
        task = asyncio.current_task()
        if task is not None:
            loaded.active_tasks.add(task)

        async def _release() -> None:
            await self.release(model_name)

        return ModelLease(
            manager=self,
            model_name=model_name,
            engine=loaded.engine,
            release_cb=_release,
        )

    async def _execute_load(self, pending: PendingLoad) -> LoadedModel:
        """Instantiate a reserved model load outside the manager lock."""
        entry = self._registry[pending.model_name]
        loaded: LoadedModel | None = None
        unload_after_load: LoadedModel | None = None

        try:
            resolved_source = await self._resolve_source(entry)
            loaded = await self._instantiate_model(entry, resolved_source)
        except Exception as exc:
            async with self._condition:
                current = self._loading.pop(pending.model_name, None)
                if current is pending and not current.future.done():
                    current.future.set_exception(exc)
                self._condition.notify_all()
            raise

        async with self._condition:
            current = self._loading.pop(pending.model_name, None)
            if current is not pending:
                unload_after_load = loaded
            elif self._shutting_down:
                unload_after_load = loaded
                if not current.future.done():
                    current.future.set_exception(
                        RuntimeError("Model manager is shutting down")
                    )
            else:
                self._loaded[pending.model_name] = loaded
                if not current.future.done():
                    current.future.set_result(loaded)
            self._condition.notify_all()

        if unload_after_load is not None:
            await unload_after_load.engine.stop()
            raise RuntimeError("Model load was aborted before it became available")

        return loaded

    async def _wait_for_change(self, timeout: float | None) -> None:
        async with self._condition:
            if timeout is None:
                await self._condition.wait()
                return
            if timeout <= 0:
                raise RuntimeError("Timed out waiting for model capacity")
            await asyncio.wait_for(self._condition.wait(), timeout=timeout)

    async def _run_unloads(self, unloads: list[LoadedModel]) -> None:
        async def _stop_engines() -> None:
            for loaded in unloads:
                try:
                    await loaded.engine.stop()
                finally:
                    async with self._condition:
                        self._unloading.pop(loaded.config.entry.name, None)
                        self._condition.notify_all()

        unload_task = asyncio.create_task(_stop_engines())
        try:
            await asyncio.shield(unload_task)
        except asyncio.CancelledError:
            with suspend_cancellation():
                await unload_task
            raise

    def _reserve_load_locked(self, model_name: str, required_bytes: int) -> PendingLoad:
        future: asyncio.Future[LoadedModel] = asyncio.get_running_loop().create_future()
        pending = PendingLoad(
            model_name=model_name,
            required_bytes=required_bytes,
            future=future,
        )
        self._loading[model_name] = pending
        return pending

    def _begin_unload_locked(self, model_name: str) -> LoadedModel:
        loaded = self._loaded.pop(model_name)
        self._unloading[model_name] = loaded
        return loaded

    def _idle_candidates_locked(
        self, *, exclude: str | None = None
    ) -> list[LoadedModel]:
        """Loaded, non-busy models eligible for eviction, oldest-used first."""
        return sorted(
            (
                loaded
                for name, loaded in self._loaded.items()
                if name != exclude and loaded.active_requests == 0
            ),
            key=lambda item: item.last_used_at,
        )

    def _reap_dead_locked(self, model_name: str) -> LoadedModel | None:
        """Evict a loaded entry whose engine died without an explicit unload.

        Only reaps idle entries (no active requests) so an in-flight lease
        is never yanked out from under its caller; a dead engine still
        holding requests gets reaped on the next acquire() once those
        requests finish and release() drops active_requests to zero.
        Returns the evicted entry (already moved into `_unloading`, ready
        for the caller to run through `_run_unloads`), or None if nothing
        needed reaping.
        """
        loaded = self._loaded.get(model_name)
        if loaded is None or loaded.active_requests > 0:
            return None
        if loaded.engine.is_healthy():
            return None
        logger.warning(
            "Model '%s' engine died without an explicit unload; reaping "
            "stale entry so it can be reloaded",
            model_name,
        )
        return self._begin_unload_locked(model_name)

    def _collect_idle_unloads_locked(
        self, requested_model: str, required_bytes: int
    ) -> list[LoadedModel]:
        selected: list[LoadedModel] = []
        projected_bytes = self._committed_bytes_locked()
        candidates = self._idle_candidates_locked(exclude=requested_model)

        for loaded in candidates:
            if projected_bytes + required_bytes <= self._config.memory_budget_bytes:
                break
            selected.append(self._begin_unload_locked(loaded.config.entry.name))
            projected_bytes -= loaded.config.estimated_memory_bytes

        return selected

    def _maybe_preempt_locked(
        self,
        *,
        model_name: str,
        required_bytes: int,
        start: float,
    ) -> set[asyncio.Task[Any]]:
        if not self._should_preempt_locked(start):
            return set()

        cancel_tasks: set[asyncio.Task[Any]] = set()
        projected_bytes = self._committed_bytes_locked()
        candidates = sorted(
            (
                loaded
                for name, loaded in self._loaded.items()
                if name != model_name and loaded.active_requests > 0
            ),
            key=lambda item: item.last_used_at,
        )

        for loaded in candidates:
            if projected_bytes + required_bytes <= self._config.memory_budget_bytes:
                break
            if loaded.preempting:
                projected_bytes -= loaded.config.estimated_memory_bytes
                continue
            loaded.preempting = True
            cancel_tasks.update(loaded.active_tasks)
            projected_bytes -= loaded.config.estimated_memory_bytes

        if cancel_tasks:
            self._condition.notify_all()
        return cancel_tasks

    def _should_wait_locked(self, start: float) -> bool:
        strategy = self._config.policy.strategy
        if strategy == "fail":
            return False
        timeout = self._remaining_wait_timeout(start)
        return timeout is None or timeout > 0

    def _should_preempt_locked(self, start: float) -> bool:
        policy = self._config.policy
        elapsed = time.monotonic() - start
        if policy.strategy == "preempt":
            return True
        if policy.strategy != "wait_then_preempt":
            return False
        trigger = policy.preempt_after_s if policy.preempt_after_s is not None else 0.0
        return elapsed >= trigger

    def _remaining_wait_timeout(self, start: float) -> float | None:
        timeout = self._config.policy.wait_timeout_s
        if timeout is None or timeout <= 0:
            return None
        return max(timeout - (time.monotonic() - start), 0.0)

    def _can_reserve_locked(self, required_bytes: int) -> bool:
        return (
            self._committed_bytes_locked() + required_bytes
            <= self._config.memory_budget_bytes
        )

    def _committed_bytes_locked(self) -> int:
        loaded_bytes = sum(
            loaded.config.estimated_memory_bytes for loaded in self._loaded.values()
        )
        loading_bytes = sum(item.required_bytes for item in self._loading.values())
        unloading_bytes = sum(
            loaded.config.estimated_memory_bytes for loaded in self._unloading.values()
        )
        return loaded_bytes + loading_bytes + unloading_bytes

    async def _instantiate_model(
        self, entry: RegisteredModel, resolved_source: str
    ) -> LoadedModel:
        config = self._resolve_model_config(entry, resolved_source)

        if self._engine_factory is not None:
            engine = self._engine_factory(config)
        else:
            # One construction path for both serving modes: the same function
            # the single-model path installs as its residency engine factory.
            # A registry entry therefore cannot drift from `serve <model>` in
            # which engine argument it passes.
            from .server import ModelSpec, _build_engine

            engine = _build_engine(
                ModelSpec(
                    model_key=config.entry.name,
                    **{
                        spec_field: getattr(config, config_attr)
                        for spec_field, config_attr in _SPEC_FROM_CONFIG.items()
                    },
                    # ModelSpec fields an entry can set that have no dedicated
                    # RegisteredModel field (today: trust_remote_code). Kept as
                    # a dict so a future ModelSpec field needs no code here.
                    **config.entry.spec_overrides,
                )
            )

        await engine.start()
        return LoadedModel(config=config, engine=engine)

    async def _resolve_source(self, entry: RegisteredModel) -> str:
        return await asyncio.to_thread(self._resolve_source_sync, entry)

    def _resolve_source_sync(self, entry: RegisteredModel) -> str:
        source = entry.source
        if Path(source).exists():
            return source
        downloaded = ensure_model_downloaded(
            source,
            config=self._defaults.download_config,
            is_mllm=is_mllm_model(source) or bool(entry.force_mllm),
        )
        return str(downloaded)

    def _resolve_estimated_bytes(
        self, entry: RegisteredModel, resolved_source: str
    ) -> int:
        if entry.estimated_memory_bytes is not None:
            return entry.estimated_memory_bytes
        estimated = _estimate_model_bytes_from_source(resolved_source)
        if estimated > 0:
            return estimated
        source_path = Path(resolved_source)
        if not source_path.exists():
            raise ValueError(
                "models-config entry "
                f"'{entry.name}' uses non-local source '{entry.source}' without "
                "estimated_memory_gb. Registry-backed loading requires an explicit "
                "memory estimate for non-local models so eviction remains deterministic."
            )

        available = _safe_available_memory_bytes()
        if available > 0:
            logger.warning(
                "Falling back to a coarse memory estimate for registry entry '%s' "
                "because no weight files were found under %s; set estimated_memory_gb "
                "explicitly for deterministic eviction.",
                entry.name,
                resolved_source,
            )
            return max(available // 8, 1)

        raise ValueError(
            "Cannot estimate memory for registry entry "
            f"'{entry.name}' from '{resolved_source}'. Set estimated_memory_gb "
            "explicitly in the models config."
        )

    def _resolve_model_config(
        self, entry: RegisteredModel, resolved_source: str
    ) -> ResolvedModelConfig:
        continuous_batching = (
            entry.continuous_batching
            if entry.continuous_batching is not None
            else self._defaults.continuous_batching
        )
        force_mllm = (
            entry.force_mllm
            if entry.force_mllm is not None
            else self._defaults.force_mllm
        )
        enable_mtp = (
            entry.enable_mtp
            if entry.enable_mtp is not None
            else self._defaults.enable_mtp
        )
        prefill_step_size = (
            entry.prefill_step_size
            if entry.prefill_step_size is not None
            else self._defaults.prefill_step_size
        )
        # A batched entry needs an explicit entry-private SchedulerConfig:
        # when the CLI serves no scheduler flags (vexillm's registry mode),
        # BatchedEngine would otherwise build its own default config and
        # drop the per-entry cache_memory_mb / prefill_step_size overrides.
        if self._defaults.scheduler_config is None and continuous_batching:
            scheduler_config = _clone_scheduler_config(
                SchedulerConfig(), prefill_step_size=prefill_step_size
            )
        else:
            scheduler_config = _clone_scheduler_config(
                self._defaults.scheduler_config, prefill_step_size=prefill_step_size
            )
        if entry.cache_memory_mb is not None:
            cache_memory_mb: int | None = entry.cache_memory_mb
        elif self._defaults.scheduler_config is not None:
            cache_memory_mb = self._defaults.scheduler_config.cache_memory_mb
        else:
            cache_memory_mb = None
        # BatchedEngine reads cache_memory_mb only from its SchedulerConfig, so a
        # per-entry override must land on the entry-private clone; the clone is
        # never shared with another entry, so the in-place set is safe.
        if entry.cache_memory_mb is not None and scheduler_config is not None:
            scheduler_config.cache_memory_mb = entry.cache_memory_mb
        if entry.scheduler:
            if scheduler_config is None:
                raise ValueError(
                    f"models-config entry '{entry.name}': "
                    f"{sorted(entry.scheduler)} are SchedulerConfig fields and "
                    "need continuous_batching: true (a simple-mode engine takes "
                    "no SchedulerConfig)"
                )
            # replace() re-runs __post_init__, so the scheduler validates these
            # exactly as it validates the CLI's own flags.
            scheduler_config = dataclasses.replace(scheduler_config, **entry.scheduler)
        if scheduler_config is not None:
            # BatchedEngine reads MTP from its SchedulerConfig; only SimpleEngine
            # takes mtp= as a constructor argument.
            scheduler_config.enable_mtp = enable_mtp
        specprefill_enabled = (
            entry.specprefill_enabled
            if entry.specprefill_enabled is not None
            else self._defaults.specprefill_enabled
        )
        specprefill_threshold = (
            entry.specprefill_threshold
            if entry.specprefill_threshold is not None
            else self._defaults.specprefill_threshold
        )
        specprefill_keep_pct = (
            entry.specprefill_keep_pct
            if entry.specprefill_keep_pct is not None
            else self._defaults.specprefill_keep_pct
        )
        specprefill_backbone_pct = (
            entry.specprefill_backbone_pct
            if entry.specprefill_backbone_pct is not None
            else self._defaults.specprefill_backbone_pct
        )
        specprefill_draft_model = (
            entry.specprefill_draft_model
            if entry.specprefill_draft_model is not None
            else self._defaults.specprefill_draft_model
        )
        prefix_trie_cache = (
            entry.prefix_trie_cache
            if entry.prefix_trie_cache is not None
            else self._defaults.prefix_trie_cache
        )
        prefix_trie_cache_size = (
            entry.prefix_trie_cache_size
            if entry.prefix_trie_cache_size is not None
            else self._defaults.prefix_trie_cache_size
        )
        prefix_trie_cache_memory_mb = (
            entry.prefix_trie_cache_memory_mb
            if entry.prefix_trie_cache_memory_mb is not None
            else self._defaults.prefix_trie_cache_memory_mb
        )
        stream_interval = (
            entry.stream_interval
            if entry.stream_interval is not None
            else self._defaults.stream_interval
        )
        gpu_memory_utilization = (
            entry.gpu_memory_utilization
            if entry.gpu_memory_utilization is not None
            else self._defaults.gpu_memory_utilization
        )
        mllm_draft_model = entry.mllm_draft_model
        mllm_draft_kind = entry.mllm_draft_kind
        mllm_draft_block_size = entry.mllm_draft_block_size
        default_mllm_draft = bool(entry.default_mllm_draft)
        reasoning_parser = (
            entry.reasoning_parser
            if entry.reasoning_parser is not None
            else self._defaults.reasoning_parser
        )
        tool_call_parser = (
            entry.tool_call_parser
            if entry.tool_call_parser is not None
            else self._defaults.tool_call_parser
        )
        enable_auto_tool_choice = (
            bool(entry.enable_auto_tool_choice)
            if entry.enable_auto_tool_choice is not None
            else self._defaults.enable_auto_tool_choice
        )
        if enable_auto_tool_choice and not tool_call_parser:
            enable_auto_tool_choice = False
        estimated_memory_bytes = self._resolve_estimated_bytes(entry, resolved_source)

        return ResolvedModelConfig(
            entry=entry,
            resolved_source=resolved_source,
            continuous_batching=continuous_batching,
            force_mllm=force_mllm,
            enable_mtp=enable_mtp,
            prefill_step_size=prefill_step_size,
            specprefill_enabled=specprefill_enabled,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
            specprefill_backbone_pct=specprefill_backbone_pct,
            specprefill_draft_model=specprefill_draft_model,
            mllm_draft_model=mllm_draft_model,
            mllm_draft_kind=mllm_draft_kind,
            mllm_draft_block_size=mllm_draft_block_size,
            default_mllm_draft=default_mllm_draft,
            prefix_trie_cache=prefix_trie_cache,
            prefix_trie_cache_size=prefix_trie_cache_size,
            prefix_trie_cache_memory_mb=prefix_trie_cache_memory_mb,
            stream_interval=stream_interval,
            gpu_memory_utilization=gpu_memory_utilization,
            scheduler_config=scheduler_config,
            estimated_memory_bytes=estimated_memory_bytes,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            enable_auto_tool_choice=enable_auto_tool_choice,
            cache_memory_mb=cache_memory_mb,
        )
