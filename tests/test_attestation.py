from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
digest_entries = runpy.run_path(str(ROOT / "scripts" / "create_attestation.py"))["digest_entries"]


def test_bundle_hash_is_order_independent_and_content_sensitive():
    entries = [("results/a.json", b"{\"a\":1}"), ("results/b.json", b"{\"b\":2}")]

    assert digest_entries(entries) == digest_entries(list(reversed(entries)))
    assert digest_entries(entries) != digest_entries([("results/a.json", b"changed"), entries[1]])
