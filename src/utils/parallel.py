from multiprocessing import Pool
from typing import Callable, Iterable

def parallel_map(
    func: Callable,
    data: Iterable,
    processes: int = None
) -> list:
    """Parallelize a function using multiprocessing."""
    with Pool(processes=processes) as pool:
        results = pool.map(func, data)
    return results