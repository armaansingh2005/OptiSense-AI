"""Report whether the local Python environment has the project dependencies."""

from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = {
    "streamlit": "streamlit",
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "Pillow",
    "pandas": "pandas",
    "plotly": "plotly",
    "pytest": "pytest",
}


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    missing: list[str] = []
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
            print(f"MISSING  {package_name}")
        else:
            print(f"OK       {package_name}")

    if missing:
        print("\nInstall missing dependencies with:")
        print("python -m pip install -r requirements.txt")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
