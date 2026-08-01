"""Persist LLM decision traces for offline SFT / analysis.

Evaluation writes one directory per run::

    datasets/llm_sft/<run_id>/
      manifest.json
      <agent_slug>/
        meta.json
        episode_000.json
        episode_001.json
        sft.jsonl          # one chat example per parse-ok step

Episode JSON holds every step (messages, market state, actions, env feedback).
``sft.jsonl`` is a flat chat corpus ready for supervised fine-tuning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

DEFAULT_DATASET_ROOT = Path("datasets") / "llm_sft"


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy / Path / datetime values into JSON-safe types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def serialize_messages(messages: list[BaseMessage] | list[Any]) -> list[dict[str, str]]:
    """Convert LangChain messages to ``[{role, content}, ...]``."""
    out: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = str(getattr(msg, "type", getattr(msg, "role", "unknown")))
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            content = "".join(parts)
        out.append({"role": role, "content": str(content)})
    return out


def slugify(label: str, *, max_len: int = 80) -> str:
    """Filesystem-safe slug from a contestant label."""
    text = label.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_") or "agent"
    return text[:max_len]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")


@dataclass
class LLMDatasetWriter:
    """Thread-friendly writer: one agent subdirectory under a shared run root."""

    root: Path
    run_id: str
    agent_label: str
    agent_name: str
    agent_params: dict[str, Any] = field(default_factory=dict)
    mode: str = "multi_asset"
    symbols: list[str] = field(default_factory=list)
    seed: int = 0
    extra_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.agent_dir = self.root / slugify(self.agent_label)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self._write_agent_meta()

    def _llm_identity(self) -> dict[str, Any]:
        llm = self.agent_params.get("llm") or {}
        if isinstance(llm, dict):
            return {
                "provider": llm.get("provider"),
                "model": llm.get("model") or llm.get("name"),
                "temperature": llm.get("temperature"),
                "reasoning": llm.get("reasoning"),
                "reasoning_effort": llm.get("reasoning_effort"),
            }
        return {"provider": None, "model": None}

    def _write_agent_meta(self) -> None:
        payload = {
            "run_id": self.run_id,
            "agent_label": self.agent_label,
            "agent_name": self.agent_name,
            "agent_params": to_jsonable(self.agent_params),
            "llm": self._llm_identity(),
            "mode": self.mode,
            "symbols": list(self.symbols),
            "seed": int(self.seed),
            **to_jsonable(self.extra_meta),
        }
        (self.agent_dir / "meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def write_episode(
        self,
        *,
        episode_index: int,
        episode_seed: int,
        steps: list[dict[str, Any]],
        episode_metrics: dict[str, float] | None = None,
    ) -> Path:
        """Write ``episode_XXX.json`` and append parse-ok steps to ``sft.jsonl``."""
        path = self.agent_dir / f"episode_{episode_index:03d}.json"
        payload = {
            "run_id": self.run_id,
            "agent_label": self.agent_label,
            "agent_name": self.agent_name,
            "llm": self._llm_identity(),
            "mode": self.mode,
            "symbols": list(self.symbols),
            "episode_index": int(episode_index),
            "episode_seed": int(episode_seed),
            "n_steps": len(steps),
            "episode_metrics": to_jsonable(episode_metrics or {}),
            "steps": to_jsonable(steps),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        sft_path = self.agent_dir / "sft.jsonl"
        with sft_path.open("a", encoding="utf-8") as fh:
            for step in steps:
                if not step.get("parse_ok"):
                    continue
                sft_messages = step.get("sft_messages") or step.get("messages")
                if not sft_messages:
                    continue
                record = {
                    "id": (
                        f"{slugify(self.agent_label)}/"
                        f"ep{episode_index:03d}/"
                        f"step{int(step.get('step', 0)):03d}"
                    ),
                    "messages": sft_messages,
                    "actions_masked": step.get("actions_masked") or {},
                    "actions_real": step.get("actions_real") or {},
                    "parse_ok": True,
                    "meta": {
                        "run_id": self.run_id,
                        "agent_label": self.agent_label,
                        "agent_name": self.agent_name,
                        "llm": self._llm_identity(),
                        "episode_index": int(episode_index),
                        "step": int(step.get("step", 0)),
                        "timestamp": step.get("timestamp"),
                        "symbol_map": step.get("symbol_map") or {},
                    },
                }
                fh.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
        return path


def write_run_manifest(
    root: Path,
    *,
    run_id: str,
    mode: str,
    symbols: list[str],
    seed: int,
    n_episodes: int,
    episode_length: int,
    contestants: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write top-level ``manifest.json`` for an evaluation dataset run."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "symbols": list(symbols),
        "seed": int(seed),
        "n_episodes": int(n_episodes),
        "episode_length": int(episode_length),
        "contestants": to_jsonable(contestants),
        "format": {
            "episode_json": "Full per-step traces (messages, state, actions, env).",
            "sft_jsonl": "One line per parse-ok step; chat messages for SFT.",
        },
        **to_jsonable(extra or {}),
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def sanitize_info(info: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe subset of env ``info`` (drops huge unused blobs if any)."""
    skip = {"asset_features"}  # features are already in market_state text; keep names
    out: dict[str, Any] = {}
    for key, value in info.items():
        if key in skip:
            continue
        out[key] = to_jsonable(value)
    # Keep compact feature matrix separately if present.
    if "asset_features" in info:
        out["asset_features"] = to_jsonable(info["asset_features"])
    if "feature_names" in info:
        out["feature_names"] = to_jsonable(info["feature_names"])
    return out
