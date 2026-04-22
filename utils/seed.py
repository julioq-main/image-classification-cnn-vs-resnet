"""
Utilities for setting global random seeds to ensure reproducibility across
PyTorch, CUDA, Python's built-in random module, and NumPy.
"""
import logging
import os
import random
from typing import Callable

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
    ``DataLoader``. See ``get_worker_init_fn`` for a compatible helper.

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


def get_worker_init_fn(seed: int | None) -> Callable[[int], None]:
    """
    Return a ``worker_init_fn`` for reproducible multi-process data loading.

    Each worker receives a unique seed derived as ``(seed + worker_id) % 2**32``,
    ensuring different but reproducible RNG states per worker. NumPy and
    Python's ``random`` module are seeded explicitly inside each worker, since
    they do not inherit the main process RNG state.

    Parameters
    ----------
    seed : int
        Base seed used to derive per-worker seeds.

    Returns
    -------
    Callable[[int], None]
        A function compatible with ``DataLoader``'s ``worker_init_fn`` parameter.

    Examples
    --------
    >>> loader = DataLoader(
    ...     dataset,
    ...     worker_init_fn=get_worker_init_fn(42),
    ...     generator=torch.Generator().manual_seed(42),
    ... )
    """
    if seed is None:
        return None
    
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn