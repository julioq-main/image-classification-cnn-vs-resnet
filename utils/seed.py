"""
utils/seed.py

Provides utilities for setting global random seeds to ensure reproducibility
across PyTorch, CUDA, Python's built-in random module, and NumPy.

Functions:
- set_seed: Sets all relevant random seeds and enforces deterministic behavior.

Notes:
- Deterministic mode may reduce performance due to the disabling of
  non-deterministic CUDA algorithms and cuDNN auto-tuning.
- Should be called once at the start of the program, before any model
  initialization, data loading, or training.
- Does not handle seeds for DataLoader worker processes. For full
  reproducibility with multiple workers, pass a worker_init_fn to DataLoader.
  See get_worker_init_fn for a compatible helper.
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Sets random seeds for all relevant libraries and enforces deterministic
    algorithm usage to ensure reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators.

    Returns:
        None

    Behavior:
        - Sets the seed for Python's built-in random module.
        - Sets the seed for NumPy's random number generator.
        - Sets the seed for PyTorch CPU and all CUDA devices.
        - Sets CUBLAS_WORKSPACE_CONFIG to ensure deterministic behavior
          for CUBLAS operations on CUDA 10.2+.
        - Disables cuDNN benchmark mode to prevent algorithm selection
          variability across runs.
        - Enables cuDNN deterministic mode for legacy compatibility.
        - Enables deterministic algorithm mode in PyTorch, which raises
          a RuntimeError if a non-deterministic operation is encountered.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Required for deterministic CUBLAS operations on CUDA 10.2+.
    # :16:8 limits workspace to 16 MB per buffer across 8 buffers.
    # Use :4096:8 instead if memory is not a concern.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # Disable cuDNN auto-tuner: prevents it from selecting the fastest
    # (but potentially non-deterministic) convolution algorithm each run.
    torch.backends.cudnn.benchmark = False

    # Explicit cuDNN deterministic mode for compatibility with older code
    # and libraries that check this flag directly.
    torch.backends.cudnn.deterministic = True

    # Force PyTorch to use deterministic algorithms only.
    # Raises RuntimeError if a non-deterministic operation is encountered.
    torch.use_deterministic_algorithms(True)

    logger.info(f"Seed set to {seed}")


def get_worker_init_fn(seed: int):
    """
    Returns a worker_init_fn to be passed to DataLoader for reproducible
    multi-process data loading.

    Args:
        seed (int): Base seed used to derive per-worker seeds.

    Returns:
        Callable: A function compatible with DataLoader's worker_init_fn
        parameter.

    Usage:
        >>> loader = DataLoader(
        ...     dataset,
        ...     worker_init_fn=get_worker_init_fn(seed),
        ...     generator=torch.Generator().manual_seed(seed)
        ... )

    Behavior:
        - Each worker receives a unique seed derived from the base seed and
          its worker ID, ensuring different but reproducible random states
          per worker.
        - Seeds both NumPy and Python's random module inside each worker
          process, since they do not inherit the main process RNG state.
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn