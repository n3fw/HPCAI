import time
import threading
from pathlib import Path

from sequential_pipeline import (
    load_image,
    process_one_image,
    compute_global_summary,
    save_results_to_csv,
)

THRESHOLD = 0.65

image_paths = sorted(Path("dataset").glob("*.npy"))
n_images = len(image_paths)

print("Number of images:", n_images)


# Experiment 1: file loading only (I/O)

def load_task(i, images):
    images[i] = load_image(image_paths[i])


# Sequential loading
images = [None] * n_images
start = time.perf_counter()

for i in range(n_images):
    load_task(i, images)

load_sequential_time = time.perf_counter() - start


# Threaded loading: one thread per image
images = [None] * n_images
threads = []
start = time.perf_counter()

for i in range(n_images):
    t = threading.Thread(
        target=load_task,
        args=(i, images),
        name=f"Load-Worker-{i}"
    )
    threads.append(t)
    t.start()

for t in threads:
    t.join()

load_thread_time = time.perf_counter() - start


# Experiment 2: complete processing (CPU-bound)

def process_task(i, results):
    # Each thread writes in its own slot of the list,
    # so no lock is needed.
    results[i] = process_one_image(image_paths[i], threshold=THRESHOLD)


# Sequential processing
results = [None] * n_images
start = time.perf_counter()

for i in range(n_images):
    process_task(i, results)

cpu_sequential_time = time.perf_counter() - start


# Threaded processing: one thread per image
results = [None] * n_images
threads = []
start = time.perf_counter()

for i in range(n_images):
    t = threading.Thread(
        target=process_task,
        args=(i, results),
        name=f"CPU-Worker-{i}"
    )
    threads.append(t)
    t.start()

for t in threads:
    t.join()

cpu_thread_time = time.perf_counter() - start


# Results

print()
print("Method                  | Time (s) | Speedup")
print(f"Loading sequential      | {load_sequential_time:.4f}   | 1.00")
print(f"Loading {n_images} threads      | {load_thread_time:.4f}   | "
      f"{load_sequential_time / load_thread_time:.2f}")
print(f"Processing sequential   | {cpu_sequential_time:.4f}   | 1.00")
print(f"Processing {n_images} threads   | {cpu_thread_time:.4f}   | "
      f"{cpu_sequential_time / cpu_thread_time:.2f}")

save_results_to_csv(results, Path("threading_results.csv"))

summary = compute_global_summary(results)
print()
print("Global summary:", summary)
