"""Re-export of src/holistic_kp.py, which lives in src/ so the submission's predict() is self-contained.
Training and evaluation scripts keep importing `holistic_kp` from here."""
import importlib.util as _u
import pathlib as _p
import sys as _s

_spec = _u.spec_from_file_location("holistic_kp", _p.Path(__file__).resolve().parents[1] / "src" / "holistic_kp.py")
_m = _u.module_from_spec(_spec)
_s.modules["holistic_kp"] = _m
_spec.loader.exec_module(_m)
