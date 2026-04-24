"""
Utilities for setting global random seeds to ensure reproducibility across
PyTorch, CUDA, Python's built-in random module, and NumPy.
"""
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for all relevant libraries and enforce deterministic behavior.

    Parameters
    ----------
    seed : int
        Seed value applied to all random number generators.

    Notes
    -----
    Enables deterministic algorithm mode in PyTorch via
    ``torch.use_deterministic_algorithms(True)``, which raises a
    ``RuntimeError`` at runtime if a non-deterministic operation is
    encountered. This may reduce performance due to the disabling of
    non-deterministic CUDA algorithms and cuDNN auto-tuning.

    Should be called once at program startup, before any model
    initialization, data loading, or training.

    Does not handle seeds for ``DataLoader`` worker processes. For full
    reproducibility with multiple workers, pass a ``worker_init_fn`` to
    ``DataLoader``. See ``worker_init_fn``.

    Examples
    --------
    >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # :16:8 limits workspace to 16 MB per buffer across 8 buffers.
    # Use :4096:8 instead if memory is not a concern.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # legacy compatibility
    torch.use_deterministic_algorithms(True)

    logger.info(f"Seed set to {seed}")


def worker_init_fn(worker_id: int, seed: int | None) -> None:
    """
    Initialize a DataLoader worker process with a deterministic random seed.

    This function is intended to be used as a PyTorch ``DataLoader.worker_init_fn``
    in combination with ``functools.partial`` to bind a fixed base seed.

    Each worker receives a unique seed derived as ``(seed + worker_id) % 2**32``

    Parameters
    ----------
    worker_id : int
        ID of the DataLoader worker process.
    seed : int
        Base seed used to derive per-worker deterministic seeds. If None,
        no seeding is applied.
    
    Notes
    -----
    This function must be defined at module top-level (not nested) to remain
    picklable under multiprocessing start methods such as ``forkserver`` or
    ``spawn`` (required in Python 3.14+ on many systems).

    It should NOT be wrapped in a closure or factory function, as that would
    break pickling for DataLoader worker processes.
        
    Examples
    --------
    Recommended usage:

    >>> from functools import partial
    >>> loader = DataLoader(
    ...     dataset,
    ...     num_workers=4,
    ...     worker_init_fn=partial(worker_init_fn, seed=42),
    ...     generator=torch.Generator().manual_seed(42),
    ... )    
    """
    if seed is None:
        return None
    
    worker_seed = (seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
