"""
Knowledge Base policies module for Naukri.com Domain Support Agent.

Provides programmatic access and utilities for loading, querying, and validating
the 12 official domain policy documents stored in policies.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

POLICIES_FILE = Path(__file__).resolve().parent / "policies.json"


def load_policies(file_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """
    Load all knowledge base policy documents from JSON.

    Args:
        file_path: Optional custom path to policies.json.

    Returns:
        List of policy dictionaries containing doc_id, title, category, content, and tags.
    """
    path = Path(file_path) if file_path else POLICIES_FILE
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data)}")

    return data


def get_policy_by_id(doc_id: str, file_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Look up a specific policy document by its doc_id (e.g. 'POL-001')."""
    policies = load_policies(file_path)
    for p in policies:
        if p.get("doc_id") == doc_id:
            return p
    return None


def get_all_policy_texts(file_path: Optional[Path | str] = None) -> List[str]:
    """Retrieve the raw content string of all policy documents."""
    policies = load_policies(file_path)
    return [p["content"] for p in policies]


if __name__ == "__main__":
    docs = load_policies()
    print(f"Successfully loaded {len(docs)} policy documents from {POLICIES_FILE}:")
    for doc in docs:
        sentences = [s.strip() for s in doc["content"].split(".") if s.strip()]
        print(f" - [{doc['doc_id']}] {doc['title']} ({len(sentences)} sentences)")
