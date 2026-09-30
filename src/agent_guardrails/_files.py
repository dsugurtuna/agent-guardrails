"""Creating state files that only their owner can read.

Why? The queue database holds the full arguments of queued actions, secrets
included, and the audit log records who did what. With a typical umask of 022
both would otherwise be readable by every user of the host. Files and
directories that already exist keep the permissions they have.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

PRIVATE_DIR = 0o700
PRIVATE_FILE = 0o600


def make_private_dir(path: Path) -> None:
    """Create ``path`` (and missing parents) if needed; a new leaf is owner-only."""
    path.mkdir(mode=PRIVATE_DIR, parents=True, exist_ok=True)


def create_private_file(path: Path) -> None:
    """Create an empty owner-only file at ``path`` unless one already exists."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE)
    except FileExistsError:
        return
    os.close(fd)


def open_private_append(path: Path) -> IO[bytes]:
    """Open ``path`` for appending and reading, creating it owner-only if needed."""
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, PRIVATE_FILE)
    try:
        return os.fdopen(fd, "a+b")
    except BaseException:
        os.close(fd)
        raise
