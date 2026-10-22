from sequential_pipeline import (
    load_image,
    process_one_image,
    mean_filter_3x3,
    compute_image_statistics,
    normalize_image
)

from machine_info import benchmark, print_benchmark_report
import argparse
import csv
import time
from pathlib import Path

import numpy as np

def normalize_image_vectorized(image: np.ndarray) -> np.ndarray:

    image_min = float(image.min())
    image_max = float(image.max())

    output = np.empty_like(image, dtype=np.float32)

    scale = image_max - image_min

    if scale == 0.0:
        output.fill(0.0)
        return output

    # Vectorisation : remplacement des deux boucles for
    output[:, :] = (image - image_min) / scale

    return output

def mean_filter_3x3_vectorized(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
   
    output = image.copy()

    total = (
        image[:-2, :-2]     # En haut à gauche
        + image[:-2, 1:-1]  # En haut
        + image[:-2, 2:]    # En haut à droite
        + image[1:-1, :-2]  # À gauche
        + image[1:-1, 1:-1] # Au centre
        + image[1:-1, 2:]   # À droite
        + image[2:, :-2]    # En bas à gauche
        + image[2:, 1:-1]   # En bas
        + image[2:, 2:]     # En bas à droite
    )
    output[1:-1, 1:-1] = total / 9.0

    return output

if __name__ == "__main__":

    image = np.load("dataset/image_0000.npy")

    # Normalisation
    results1 = benchmark(normalize_image, image, repeat=10, warmup=1)
    results2 = benchmark(normalize_image_vectorized, image, repeat=10, warmup=1)

    print("Normalize sequential:")
    print_benchmark_report(results1)

    print("Normalize vectorized:")
    print_benchmark_report(results2)

    print("Speedup:", results1[3] / results2[3])

    # Mean filter
    normalized = normalize_image(image)

    results3 = benchmark(mean_filter_3x3, normalized, repeat=10, warmup=1)
    results4 = benchmark(mean_filter_3x3_vectorized, normalized, repeat=10, warmup=1)

    print("Mean filter sequential:")
    print_benchmark_report(results3)

    print("Mean filter vectorized:")
    print_benchmark_report(results4)

    print("Speedup:", results3[3] / results4[3])

    print("Normalization correct:",
      np.allclose(normalize_image(image), normalize_image_vectorized(image)))

    print("Mean filter correct:",
        np.allclose(mean_filter_3x3(normalized),
                    mean_filter_3x3_vectorized(normalized)))
