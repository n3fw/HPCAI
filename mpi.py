from mpi4py import MPI
import time
import numpy as np
from pathlib import Path
from sequential_pipeline import (
    process_one_image, 
    save_results_to_csv
)

global THRESHOLD
THRESHOLD = 0.65

def chunk_list(lst, n):
    """Divise une liste 'lst' en 'n' sous-listes de tailles égales."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunks = None
    counter_img = 0
    times = []

    if rank == 0:
        images = sorted(Path("dataset").glob("*.npy"))
        chunks = chunk_list(images, size)
    mes_images = comm.scatter(chunks, root=0)
    print(f"Rank {rank}: {len(mes_images)} recieved")

    results = []
    for img in mes_images:
        start = time.perf_counter()
        res = process_one_image(img, THRESHOLD)
        times.append(time.perf_counter() - start)
        results.append(res)
        counter_img += 1

    results = comm.gather(results, root=0)
    if rank == 0:
        final_results = [item for sublist in results for item in sublist]
        save_results_to_csv(final_results, Path("results/mpi_results.csv"))
        print("Cumulative time :", np.sum(times))
        print("Speedup is of about", 6.03 / np.sum(times))

    

if __name__ == "__main__":
    times = main()