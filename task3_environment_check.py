
import sys
from pathlib import Path
import importlib.util

print("Python:", sys.version)
print("Executable:", sys.executable)

for package in ["pandas", "numpy", "sklearn", "psutil"]:
    installed = importlib.util.find_spec(package) is not None
    print(f"{package}: {'Installed' if installed else 'Not installed'}")

try:
    import psutil

    memory = psutil.virtual_memory()
    print(f"\nTotal RAM: {memory.total / (1024**3):.2f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
except ImportError:
    print("\nInstall psutil to check available RAM.")

base = Path(__file__).resolve().parent
for filename in [
    "output/splits/train.csv",
    "output/splits/test.csv",
]:
    path = base / filename
    if path.exists():
        print(f"{filename}: {path.stat().st_size / (1024**3):.2f} GB")
    else:
        print(f"MISSING: {filename}")
