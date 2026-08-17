from pathlib import Path
from datetime import datetime

def get_repo_root() -> Path:
    """Obtains the path to the root of the github repository.

    Walks up the parents until an entry named `.git` is found.

    `.git` is tested with `exists()` rather than `is_dir()` on purpose. In a
    linked worktree (`git worktree add`) and in a submodule, `.git` is a *file*
    holding a `gitdir:` pointer, not a directory. Requiring a directory made this
    raise inside every worktree, which took out 7 of 29 tests — including the
    end-to-end pipeline test — for anyone running from one.

    Returns:
        pathlib.Path: The path to the repository root.

    Raises:
        FileNotFoundError: If no .git entry is found.
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / '.git').exists():
            return parent
    raise FileNotFoundError("Could not find git repository root.")

def get_current_time_str() -> str:
    """Returns the current time in YYYYMMDD-HHMMSS format.
    
    Returns:
        str: Current time formatted as YYYYMMDD-HHMMSS
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")

