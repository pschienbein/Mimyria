import sys
import importlib

PKG = "mimyria.scripts"


def main():
    if len(sys.argv) < 2:
        print("Usage: mimyria-ml <command> [args]")
        return 1

    cmd = sys.argv[1]

    try:
        mod = importlib.import_module(f"{PKG}.{cmd}")
    except ModuleNotFoundError:
        print(f"Unknown command: {cmd}")
        return 2

    if not hasattr(mod, "main"):
        print(f"{cmd}: missing main")
        return 3

    return mod.main(sys.argv[2:])
