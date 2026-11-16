"""Install version-matched acados/MSVC patches, checking all files before changes."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile


def digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_dir.resolve()
    package = Path(__file__).resolve().parent
    if not (root / "pyproject.toml").is_file():
        raise SystemExit("Run from the Leap-C repository root containing pyproject.toml.")
    manifest = json.loads((package / "windows_fix_v2_manifest.json").read_text(encoding="utf-8"))
    pending = []
    for item in manifest["files"]:
        target = root / item["target"]
        payload = package / "payload" / item["payload"]
        if not target.is_file() or not payload.is_file():
            raise SystemExit(f"Missing source or payload: {target} / {payload}")
        if digest(payload) != item["sha256"]:
            raise SystemExit(f"Patch payload checksum mismatch: {payload}")
        current = digest(target)
        if current == item["sha256"]:
            print(f"Already patched: {item['target']}", flush=True)
            continue
        if current not in item["accepted_sha256"]:
            raise SystemExit(f"Different source version; no files changed: {target}")
        backup = target.with_name(target.name + ".before_leapc_windows_fix_v2.bak")
        if backup.exists() and digest(backup) not in item["accepted_sha256"]:
            raise SystemExit(f"Existing backup has different content: {backup}")
        pending.append((target, payload, backup))

    for target, payload, backup in pending:
        if not backup.exists():
            shutil.copy2(target, backup)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix="leapc_msvc_", delete=False) as stream:
                staged = Path(stream.name)
                stream.write(payload.read_bytes())
            os.replace(staged, target)
        finally:
            if staged is not None and staged.exists():
                staged.unlink()
        print(f"Patched: {target}", flush=True)
        print(f"Backup:  {backup}", flush=True)
    print("Patch v2 installation complete. Rebuild/install the acados core before testing Leap-C.", flush=True)


if __name__ == "__main__":
    main()
