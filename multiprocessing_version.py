import os
import time
from multiprocessing import Pool
from pathlib import Path

from sequential_pipeline import (
    process_one_image,
    compute_global_summary,
    save_results_to_csv,
)

THRESHOLD = 0.65


def worker(path):
    # Each worker process handles one complete image.
    return process_one_image(path, threshold=THRESHOLD)


if __name__ == "__main__":

    image_paths = sorted(Path("dataset").glob("*.npy"))

    print("Logical CPUs reported by Python:", os.cpu_count())
    print("Number of images:", len(image_paths))
    print()

    times = {}

    for p in [1, 2, 4, 8, 12]:

        start = time.perf_counter()

        with Pool(processes=p) as pool:
            # map() distributes the images among the worker processes
            # and returns the results in input order.
            results = pool.map(worker, image_paths)

        T_p = time.perf_counter() - start
        times[p] = T_p

        print(f"{p} process(es): {T_p:.3f} s")

    print()
    print("p | Tp (s) | Speedup | Efficiency")
    for p, T_p in times.items():
        speedup = times[1] / T_p
        efficiency = speedup / p
        print(f"{p} | {T_p:.3f} | {speedup:.2f} | {efficiency:.2f}")

    # The results are the same for every p: we save the last ones.
    save_results_to_csv(results, Path("multiprocessing_results.csv"))

    summary = compute_global_summary(results)
    print()
    print("Global summary:", summary)
