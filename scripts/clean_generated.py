from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "results", ROOT / "models"):
    for path in folder.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            import shutil

            shutil.rmtree(path)
print("Removed generated results, plots, thresholds, and model artifacts.")
