"""Bootstrap the user's already installed Leap-C / bundled acados environment."""
from pathlib import Path
import hashlib
import json
import os
import sys

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "data"
ROOT = PACKAGE.parent
_DLL_HANDLES = []


def configure(require_solver=True):
    import yaml
    settings = yaml.safe_load((PACKAGE / "settings.yaml").read_text(encoding="utf-8"))
    if require_solver:
        # Prefer the bundled acados in the same Leap-C checkout, never the latch env's copy.
        acados = ROOT / "external" / "acados"
        if not (acados / "CMakeLists.txt").is_file():
            raise RuntimeError(
                "Extract latch_leapc into your Leap-C repository root, next to leap_c and external. "
                f"Expected: {acados}"
            )
        os.environ["ACADOS_SOURCE_DIR"] = str(acados)
        os.environ["PATH"] = os.pathsep.join(
            [str(acados / "bin"), str(acados / "lib"), os.environ.get("PATH", "")]
        )
        if os.name == "nt":
            required = [acados / "bin/acados.dll", acados / "include/acados_c/ocp_nlp_interface.h"]
            if any(not p.is_file() for p in required):
                raise RuntimeError("The bundled acados core is not installed. Use your working v2 build.")
            for p in (acados / "bin", acados / "lib"):
                if p.is_dir():
                    _DLL_HANDLES.append(os.add_dll_directory(str(p)))
            if " " in str(ROOT):
                raise RuntimeError("Place the Leap-C checkout in a path without spaces for CMake.")
        tmp = ROOT / "latch_leapc_tmp"
        tmp.mkdir(exist_ok=True)
        os.environ["TEMP"] = os.environ["TMP"] = str(tmp)
        import tempfile
        tempfile.tempdir = str(tmp)
        import acados_template
        loaded = Path(acados_template.__file__).resolve()
        if not loaded.is_relative_to((acados / "interfaces/acados_template").resolve()):
            raise RuntimeError(f"Wrong acados_template: {loaded}. Activate the leapc environment.")
        import leap_c
        print(f"Python: {sys.executable}\nLeap-C: {leap_c.__file__}\nacados: {loaded}", flush=True)
    return settings


def parameters():
    import yaml
    cfg = DATA / "EDGAR_Frenet"
    return yaml.safe_load((cfg / "MPC_params_frenet.yaml").read_text()), cfg


def fingerprint(settings):
    """Invalidate both binary caches and datasets after model/config changes."""
    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml", ".csv", ".json"}:
            h.update(str(path.relative_to(PACKAGE)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def workspace():
    out = ROOT / "latch_leapc_results"
    out.mkdir(exist_ok=True)
    return out
