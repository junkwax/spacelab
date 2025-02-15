from mpi4py import MPI

def parallel_map(func, data):
    """Parallelize a function across MPI ranks."""
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    results = []
    for i, item in enumerate(data):
        if i % size == rank:
            results.append(func(item))
    return comm.allreduce(results)