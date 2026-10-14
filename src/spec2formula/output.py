"""Exclusive output reservation and atomic publication without replacement."""
from contextlib import contextmanager
from pathlib import Path
import os
import warnings


@contextmanager
def exclusive_output(path):
    """Reserve .partial before work; retain it on failure for inspection.

    The final hard link creates the destination atomically and fails if any
    filesystem entry already occupies it. Both paths share one directory.
    A filesystem without hard-link support fails rather than replacing data.
    """
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f'Output already exists: {path}')
    # FileExistsError here does not give us ownership of another job's file.
    with partial.open('x', encoding='utf-8') as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    try:
        partial.unlink()
    except OSError as exc:
        warnings.warn(f'Output saved to {path}; could not remove {partial}: {exc}', RuntimeWarning)
