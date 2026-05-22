
import subprocess
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parents[0]


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
                cmd,
                cwd=REPO_DIR,
                text=True,
                stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unknown'


def _live_git_info():
    if not (REPO_DIR / ".git").exists():
        raise RuntimeError("not a git checkout")

    return {
        "commit": _git(['git', 'rev-parse', '--short', 'HEAD']),
        "branch": _git(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        "describe": _git(['git', 'describe', '--tags', '--dirty', '--always'])
    }


def _baked_git_info():
    try:
        from ._version import __version__
    except Exception:
        __version__ = 'unknown'

    return {
            'commit': '-',
            'branch': 'release',
            'describe': __version__,
        }


def git_info():
    try:
        return _live_git_info()
    except Exception:
        return _baked_git_info()


def git_info_string():
    info = git_info()
    return f'{info["describe"]} (Commit {info["commit"]} @ {info["branch"]} branch)'
