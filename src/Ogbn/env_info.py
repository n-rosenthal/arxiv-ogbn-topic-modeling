"""Print the software/hardware facts the paper's "Configuração experimental" needs.

    python -m src.Ogbn.env_info
"""
import platform
import subprocess
from importlib import metadata


def _run(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    print("python", platform.python_version(), "|", platform.platform())
    for pkg in ("torch", "scikit-learn", "sentence-transformers", "numpy", "scipy", "pandas", "transformers"):
        try:
            print(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            print(f"{pkg} (not installed)")
    try:
        import torch
        print("cuda available:", torch.cuda.is_available(), "| threads:", torch.get_num_threads())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
    except ImportError:
        pass
    print("cpu:", _run("lscpu | grep 'Model name' | sed 's/.*: *//'"), "| cores:", _run("nproc"))
    print("ram:", _run("free -h | awk '/Mem:/{print $2}'"))
