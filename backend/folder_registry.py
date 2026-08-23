"""Which local folder is the active watched folder, and which ones we have seen.

Stored beside the Chroma database rather than inside a watched folder: it
describes the store, and a watched folder may be read-only or unplugged.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

REGISTRY_NAME = "folders.json"


def registry_path(chroma_path: str) -> Path:
    p = Path(chroma_path)
    p.mkdir(parents=True, exist_ok=True)
    return p / REGISTRY_NAME


def load_registry(chroma_path: str) -> Dict[str, Any]:
    path = registry_path(chroma_path)
    default: Dict[str, Any] = {"active": None, "known": []}
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return default
    if not isinstance(data, dict):
        return default
    active = data.get("active")
    known = data.get("known")
    return {
        "active": active if isinstance(active, str) else None,
        "known": [k for k in known if isinstance(k, str)] if isinstance(known, list) else [],
    }


def save_registry(chroma_path: str, reg: Dict[str, Any]) -> None:
    registry_path(chroma_path).write_text(json.dumps(reg, indent=2, sort_keys=True))


def active_folder(chroma_path: str, fallback: str) -> str:
    return load_registry(chroma_path)["active"] or fallback


def known_folders(chroma_path: str, fallback: str) -> List[str]:
    known = list(load_registry(chroma_path)["known"])
    if not known and fallback:
        known.append(fallback)
    return sorted(known)


def set_active(chroma_path: str, folder: str) -> Dict[str, Any]:
    resolved = str(Path(folder).expanduser().resolve())
    reg = load_registry(chroma_path)
    reg["active"] = resolved
    if resolved not in reg["known"]:
        reg["known"].append(resolved)
    save_registry(chroma_path, reg)
    return reg
