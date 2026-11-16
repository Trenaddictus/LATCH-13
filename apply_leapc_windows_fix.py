"""Apply the two version-matched patches, keeping backups of the originals."""

from pathlib import Path
import argparse
import hashlib
import os
import shutil
import tempfile


PATCHES = [
    (
        "external/acados/interfaces/acados_template/acados_template/acados_ocp_batch_solver.py",
        "acados_ocp_batch_solver.py",
        "21844d1227b590a3e5f6d9cd4652590e4245f16cfc97949c54058bf74d7d8658",
        "77b813ac7bd73ae13e97f902917325f59bfe65c443a2455c4a9daca55b7f2af7",
    ),
    (
        "leap_c/diff_mpc/utils/create_solver.py",
        "create_solver.py",
        "4d28163e6a6e3e6adc303cf8161d8aecf32765ce583a0e051ee1809dc1b31c08",
        "29c88a31033228dfec5ee7f45806a8c3d40e6bec4727af0c14af4f5c3d3957ad",
    ),
]


def digest(path):
    # Treat CRLF and LF checkouts as the same Python source.
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_dir.resolve()
    if not (root / "pyproject.toml").is_file():
        raise SystemExit("Run from the Leap-C repository root containing pyproject.toml.")
    if any(character.isspace() for character in str(root)):
        raise SystemExit("Use a repository path without whitespace for this MSVC patch.")

    payload_dir = Path(__file__).resolve().parent / "patches"
    pending = []
    # Check BOTH targets before making changes to either file.
    for relative, filename, original_hash, patched_hash in PATCHES:
        target = root / relative
        payload = payload_dir / filename
        if not target.is_file() or not payload.is_file():
            raise SystemExit(f"Missing target or patch payload: {target} / {payload}")
        if digest(payload) != patched_hash:
            raise SystemExit(f"Patch payload does not match this package: {payload}")
        current = digest(target)
        if current == patched_hash:
            print(f"Already patched: {relative}", flush=True)
            continue
        if current != original_hash:
            raise SystemExit(
                f"Different source version; no files were changed: {target}\n"
                "Send the current file for a matching patch."
            )
        backup = target.with_name(target.name + ".before_leapc_windows_fix.bak")
        if backup.exists() and digest(backup) != original_hash:
            raise SystemExit(f"An existing backup has different content: {backup}")
        pending.append((target, payload, backup))

    for target, payload, backup in pending:
        if not backup.exists():
            shutil.copy2(target, backup)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix="leapc_patch_", delete=False) as stream:
                staged = Path(stream.name)
                stream.write(payload.read_bytes())
            os.replace(staged, target)
        finally:
            if staged is not None and staged.exists():
                staged.unlink()
        print(f"Patched: {target}", flush=True)
        print(f"Backup:  {backup}", flush=True)
    print("Patch installation complete. Restart Python before testing.", flush=True)


if __name__ == "__main__":
    main()
