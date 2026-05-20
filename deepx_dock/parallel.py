"""
Parallel processing utilities.

This module provides a parallel map interface supporting two backends:
- ThreadPoolExecutor: better memory efficiency and faster startup.
- joblib: suitable for CPU-bound tasks that hold the GIL.
"""

from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from typing import Callable, Iterable, TypeVar, List
from tqdm import tqdm

from deepx_dock.CONSTANT import PARALLEL_BACKEND_DEFAULT
from deepx_dock.CONSTANT import PARALLEL_BACKEND_VALID

T = TypeVar("T")
R = TypeVar("R")



def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_jobs: int = -1,
    desc: str = "Processing",
    backend: str = PARALLEL_BACKEND_DEFAULT,
) -> List[R]:
    """
    Parallel map with selectable backend.

    Parameters
    ----------
    func : Callable[[T], R]
        Function to apply to each item. Should be thread-safe when using
        the 'thread' backend.
    items : Iterable[T]
        Items to process. Will be converted to a list internally.
    n_jobs : int, optional
        Number of parallel jobs:
        - -1: Auto-detect (use all available cores)
        - 1: Sequential execution
        - N > 1: Parallel execution by N workers
        Default: -1
    desc : str, optional
        Description for progress bar. Default: "Processing"
    backend : str, optional
        Parallelization backend:
        - 'thread': ThreadPoolExecutor (default, memory-efficient,
          ideal for numpy/h5py/scipy that release the GIL)
        - 'joblib': joblib.Parallel (for CPU-bound tasks holding the GIL)
        Default: PARALLEL_BACKEND_DEFAULT (defined in CONSTANT.py)

    Returns
    -------
    List[R]
        List of results in the same order as input items.

    Raises
    ------
    ValueError
        If backend is not 'thread' or 'joblib'.
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
    >>> results = parallel_map(process, range(10), n_jobs=4, backend='joblib')

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
    total = len(items) if hasattr(items, '__len__') else None

    if n_jobs == 1:
        # Sequential execution (faster for small tasks, no threading overhead)
        return [func(item) for item in tqdm(items, total=total, desc=desc)]

    if "thread" == backend:
        max_workers = None if n_jobs < 0 else n_jobs
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                tqdm(executor.map(func, items), total=total, desc=desc)
            )
    elif "joblib" == backend:
        return Parallel(n_jobs=n_jobs, return_as="list")(
            delayed(func)(item)
            for item in tqdm(items, total=total, desc=desc)
        )
    else:
        raise ValueError(f"Unknown parallel backend: {backend}. Choose from {PARALLEL_BACKEND_VALID}.")

