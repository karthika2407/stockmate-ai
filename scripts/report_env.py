"""Print what the build is actually running on.

Reproducing a CI failure means knowing the interpreter, the platform and the
versions of the libraries that carry native code. Printing them as a step of
its own means a failed run says which of them was missing.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGES = [
    "numpy", "pandas", "sklearn", "statsmodels", "pulp",
    "cv2", "onnxruntime", "fastapi", "pydantic", "pytest",
]


def main() -> int:
    print(f"python   {platform.python_version()} ({sys.executable})")
    print(f"platform {platform.platform()}")
    print(f"machine  {platform.machine()}")

    for name in PACKAGES:
        try:
            module = __import__(name)
        except Exception as exc:
            print(f"  {name:<14} IMPORT FAILED: {type(exc).__name__}: {exc}")
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"  {name:<14} {version}")

    import stockmate

    print(f"  {'stockmate':<14} {stockmate.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
