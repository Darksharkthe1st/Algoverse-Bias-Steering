from pathlib import Path
from datetime import datetime

def get_repo_root() -> Path:
    """Obtains the path to the root of the github repository.

    Walks parents (closest first) until one contains a `.git` entry. We check
    for existence rather than a directory specifically: in a linked git worktree
    the worktree root holds a `.git` *file* (a gitdir pointer), not a directory,
    so an `.is_dir()` check would skip it and wrongly resolve to the main
    checkout's root.

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

