"""Re-export of src/extract_v2.py, which lives in src/ so the submission's predict() is self-contained.
Training and evaluation scripts keep importing `extract_v2` from here."""
import importlib.util as _u
import pathlib as _p
import sys as _s

_spec = _u.spec_from_file_location("extract_v2", _p.Path(__file__).resolve().parents[1] / "src" / "extract_v2.py")
_m = _u.module_from_spec(_spec)
_s.modules["extract_v2"] = _m
_spec.loader.exec_module(_m)
if __name__ == "__main__":
    _m.main()
