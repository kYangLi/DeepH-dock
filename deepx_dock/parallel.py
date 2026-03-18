"""
Parallel processing utilities using ThreadPoolExecutor.

This module provides a simple and efficient parallel processing interface
using Python's standard library ThreadPoolExecutor. It's designed to replace
joblib.Parallel for better memory efficiency and faster startup.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar, List
from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_jobs: int = -1,
    desc: str = "Processing",
) -> List[R]:
    """
    Parallel map using ThreadPoolExecutor.

    This function provides a simple interface for parallel processing,
    similar to joblib.Parallel but using ThreadPoolExecutor for better
    memory efficiency and faster startup.

    Parameters
    ----------
    func : Callable[[T], R]
        Function to apply to each item. Should be thread-safe.
    items : Iterable[T]
        Items to process. Will be converted to a list internally.
    n_jobs : int, optional
        Number of parallel jobs:
        - -1: Auto-detect (use all available cores)
        - 1: Sequential execution (no threading overhead)
        - N > 1: Use N threads
        Default: -1
    desc : str, optional
        Description for progress bar. Default: "Processing"

    Returns
    -------
    List[R]
        List of results in the same order as input items.

    Raises
    ------
    Exception
        Any exception raised by func will be propagated.

    Examples
    --------
    Basic usage:

    >>> from deepx_dock.parallel import parallel_map
    >>> def process(x):
    ...     return x ** 2
    >>> results = parallel_map(process, range(10), n_jobs=4)
    >>> results
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    With numpy operations (thread-safe, releases GIL):

    >>> import numpy as np
    >>> def compute_eigenvalues(n):
    ...     matrix = np.random.rand(n, n)
    ...     return np.linalg.eigvals(matrix)
    >>> results = parallel_map(compute_eigenvalues, [100, 100, 100], n_jobs=-1)

    Notes
    -----
    - ThreadPoolExecutor uses threads, which share memory. This is more
      memory-efficient than multiprocessing but requires thread-safe operations.
    - numpy, scipy, h5py operations are thread-safe and release the GIL,
      making them ideal for ThreadPoolExecutor.
    - For n_jobs=1, sequential execution is used to avoid threading overhead.
    - For n_jobs=-1, ThreadPoolExecutor's default behavior is used, which
      typically uses CPU_cores * 5 threads.

    See Also
    --------
    concurrent.futures.ThreadPoolExecutor : The underlying executor.
    """
    items = list(items)  # Convert to list for tqdm total

    if n_jobs == 1:
        # Sequential execution (faster for small tasks, no threading overhead)
        return [func(item) for item in tqdm(items, desc=desc)]

    # Parallel execution with ThreadPoolExecutor
    max_workers = None if n_jobs < 0 else n_jobs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(func, items), total=len(items), desc=desc))

    return results
