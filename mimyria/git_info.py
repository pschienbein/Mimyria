
import subprocess
from pathlib import Path

def git(cmd: list[str]) -> str:
    try:
        REPO_DIR = Path(__file__).resolve().parent
        return subprocess.check_output(cmd, cwd=REPO_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unknown'


def git_info():
    commit = git(['git', 'rev-parse', '--short', 'HEAD'])
    branch = git(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    describe = git(['git', 'describe', '--tags', '--dirty', '--always'])

    return {'commit': commit, 'branch': branch, 'describe': describe }


def git_info_string():
    info = git_info()
    return f'{info["describe"]} (Commit {info["commit"]} @ {info["branch"]} branch)'
