import sys
import importlib
import pkgutil

from mimyria.git_info import git_info_string

PKG = "mimyria.scripts"

def available_commands():
    try:
        pkg = importlib.import_module(PKG)
    except ModuleNotFoundError:
        return []

    cmds = []

    for _, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if ispkg:
            continue

        try:
            mod = importlib.import_module(f"{PKG}.{name}")
            if hasattr(mod, "main"):
                cmds.append(name)
        except Exception:
            pass

    return sorted(cmds)


def print_help():
    cmds = available_commands()

    print("Usage: mimyria-py <command> [args]\n")

    if cmds:
        print("Available Commands:")
        for c in cmds:
            print(f"  {c}")

        print('')
        print('Run "mimyria-py <command> --help" to get detailed information on each command\n')


def main():
    # Print Header
    print(f'This is mimyria-py,', git_info_string())

    if len(sys.argv) < 2:
        print_help()
        return 1

    cmd = sys.argv[1]

    if cmd in ("help", "-h", "--help"):
        print_help()
        return 0

    try:
        mod = importlib.import_module(f"{PKG}.{cmd}")
    except ModuleNotFoundError:
        print(f"Unknown command: {cmd}")
        print_help()
        return 2

    if not hasattr(mod, "main"):
        print(f"{cmd}: missing main")
        print_help()
        return 3

    return mod.main(sys.argv[2:])
