
import subprocess
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parents[0]


def _read_git_head(repo_dir: Path):
    git_dir = repo_dir / ".git"

    # normal checkout: .git is a directory
    if git_dir.is_dir():
        git_path = git_dir

    # worktree/submodule: .git is a file containing "gitdir: ..."
    elif git_dir.is_file():
        text = git_dir.read_text().strip()
        if not text.startswith("gitdir:"):
            raise RuntimeError("invalid .git file")
        git_path = (repo_dir / text.split(":", 1)[1].strip()).resolve()

    else:
        raise RuntimeError("not a git checkout")

    head = (git_path / "HEAD").read_text().strip()

    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1]
        branch = ref.removeprefix("refs/heads/")
        ref_file = git_path / ref

        if ref_file.exists():
            commit = ref_file.read_text().strip()
        else:
            commit = _read_packed_ref(git_path, ref)

        return branch, commit

    # detached HEAD
    return "HEAD", head


def _read_packed_ref(git_path: Path, ref: str):
    packed_refs = git_path / "packed-refs"

    if not packed_refs.exists():
        return "unknown"

    with packed_refs.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("^"):
                continue

            commit, name = line.split(" ", 1)
            if name == ref:
                return commit

    return "unknown"


def _manual_git_info():
    branch, commit = _read_git_head(REPO_DIR)

    short_commit = commit[:7] if commit != "unknown" else "unknown"

    return {
        "commit": short_commit,
        "branch": branch,
        "describe": short_commit,
    }


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(
            cmd,
            cwd=REPO_DIR,
            text=True,
            stderr=subprocess.DEVNULL).strip()


def _live_git_info():
    if not (REPO_DIR / ".git").exists():
        raise RuntimeError("not a git checkout")

    # for the special case that compute nodes dont have git installed: 
    try:
        return {
            "commit": _git(['git', 'rev-parse', '--short', 'HEAD']),
            "branch": _git(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
            "describe": _git(['git', 'describe', '--tags', '--dirty', '--always'])
        }

    except FileNotFoundError:
        return _manual_git_info()


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
