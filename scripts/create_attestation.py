from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "marg-attestation.json"
SOURCE_ROOTS = ("src/", "bench/", "tests/", "scripts/", "contracts/", "web/")
SOURCE_FILES = {"README.md", "DESIGN.md", "Makefile", "pyproject.toml", "foundry.toml"}
IGNORED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_entries(entries: list[tuple[str, bytes]]) -> str:
    """Hash ordered path/content pairs with separators to prevent ambiguous bundles."""
    digest = hashlib.sha256()
    for name, content in sorted(entries):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def read_paths(paths: list[Path]) -> list[tuple[str, bytes]]:
    return [(path.relative_to(ROOT).as_posix(), path.read_bytes()) for path in paths]


def create_attestation() -> dict[str, object]:
    result_paths = sorted((ROOT / "results").glob("*.json"))
    if not result_paths:
        raise FileNotFoundError("No results/*.json files found; run the benchmarks first.")

    source_entries: list[tuple[str, bytes]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in IGNORED_DIRS for part in path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in SOURCE_FILES or relative.startswith(SOURCE_ROOTS):
            source_entries.append((relative, path.read_bytes()))

    result_entries = read_paths(result_paths)
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "result_files": {name: sha256(content) for name, content in result_entries},
        "result_digest": digest_entries(result_entries),
        "source_digest": digest_entries(source_entries),
        "source_files": len(source_entries),
        "note": "Hashes verify bytes only; they do not prove benchmark correctness or production validity.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash Marg source and saved benchmark JSON for an optional EVM receipt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail unless the saved manifest matches current files.")
    args = parser.parse_args()
    expected = json.dumps(create_attestation(), indent=2, sort_keys=True) + "\n"
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.check:
        if not output.is_file() or output.read_text() != expected:
            raise SystemExit(f"{output} is stale; run `make attest` after committing source and results.")
        print(f"Verified {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected)
    manifest = json.loads(expected)
    print(f"Wrote {output}")
    print(f"Result digest: 0x{manifest['result_digest']}")
    print(f"Source digest: 0x{manifest['source_digest']}")


if __name__ == "__main__":
    main()
