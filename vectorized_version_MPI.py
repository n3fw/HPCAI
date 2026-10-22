from sequential_pipeline import (
    load_image,
    mean_filter_3x3,
    compute_image_statistics,
    normalize_image,
    compute_global_summary,
    save_results_to_csv
)


from machine_info import benchmark, print_benchmark_report
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

def compute_image_statistics_vectorized(
    image: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    height, width = image.shape

    # Vectorisation : remplacement des deux boucles for
    # Somme en float64 pour garder la même précision que le baseline
    total = float(image.sum(dtype=np.float64))
    maximum = float(image.max())
    above_threshold = int(np.count_nonzero(image >= threshold))

    pixel_count = height * width

    mean_value = total / pixel_count
    fraction_above_threshold = above_threshold / pixel_count

    return {
        "mean": mean_value,
        "max": maximum,
        "above_threshold": float(above_threshold),
        "fraction_above_threshold": fraction_above_threshold,
    }

def run_vectorized_pipeline(
    data_dir: Path,
    output_csv: Path,
    threshold: float,
) -> None:
    """
    Run the complete application with vectorized functions.
    """

    image_paths = sorted(data_dir.glob("*.npy"))

    if not image_paths:
        raise FileNotFoundError(
            f"No .npy files were found in '{data_dir}'.\n"
            "Generate the demo dataset first with:\n"
            "    python sequential_pipeline.py --generate-demo"
        )

    print("=" * 68)
    print("Vectorized scientific image-processing pipeline")
    print("=" * 68)
    print(f"Dataset directory : {data_dir}")
    print(f"Number of images  : {len(image_paths)}")
    print(f"Threshold         : {threshold}")
    print()

    results: list[dict[str, float | str]] = []

    start_time = time.perf_counter()

    for index, path in enumerate(image_paths, start=1):

        # Chargement de l'image
        image = load_image(path)

        # Normalisation vectorisée
        normalized = normalize_image_vectorized(image)

        # Filtre 3x3 vectorisé
        filtered = mean_filter_3x3_vectorized(normalized)

        # Statistiques vectorisées
        statistics = compute_image_statistics_vectorized(
            filtered,
            threshold=threshold,
        )

        result = {
            "file": path.name,
            **statistics,
        }

        results.append(result)

        print(
            f"[{index:03d}/{len(image_paths):03d}] "
            f"{path.name} processed"
        )

    elapsed_time = time.perf_counter() - start_time

    summary = compute_global_summary(results)

    save_results_to_csv(
        results,
        output_path=output_csv,
    )

    print()
    print("=" * 68)
    print("Global summary")
    print("=" * 68)
    print(
        f"Mean of image means            : "
        f"{summary['mean_of_means']:.8f}"
    )
    print(
        f"Global maximum                 : "
        f"{summary['global_max']:.8f}"
    )
    print(
        f"Total pixels above threshold   : "
        f"{summary['total_above_threshold']:.0f}"
    )
    print(
        f"Average fraction above threshold: "
        f"{summary['average_fraction_above_threshold']:.8f}"
    )
    print()
    print(f"Vectorized execution time      : {elapsed_time:.6f} s")
    print(f"Results written to             : {output_csv}")
    print("=" * 68)

def process_one_image(
    path: Path,
    threshold: float,
) -> dict[str, float | str]:
    """
    Complete sequential processing pipeline for one image.

    Pipeline:
        load
          -> normalize
          -> 3 x 3 mean filter
          -> statistics
    """

    image = load_image(path)

    normalized = normalize_image_vectorized(image)

    filtered = mean_filter_3x3_vectorized(normalized)

    statistics = compute_image_statistics_vectorized(
        filtered,
        threshold=threshold,
    )

    return {
        "file": path.name,
        **statistics,
    }
if __name__ == "__main__":

    image = np.load("dataset/image_0000.npy")

    # Normalisation
    results1 = benchmark(normalize_image, image, repeat=10, warmup=1)
    results2 = benchmark(normalize_image_vectorized, image, repeat=10, warmup=1)

    print("Normalize sequential:")
    print_benchmark_report(results1)
    print("\n")

    print("Normalize vectorized:")
    print_benchmark_report(results2)

    print("Speedup:", results1[3] / results2[3])
    print("\n")

    # Mean filter
    normalized = normalize_image(image)

    results3 = benchmark(mean_filter_3x3, normalized, repeat=10, warmup=1)
    results4 = benchmark(mean_filter_3x3_vectorized, normalized, repeat=10, warmup=1)

    print("Mean filter sequential:")
    print_benchmark_report(results3)
    print("\n")

    print("Mean filter vectorized:")
    print_benchmark_report(results4)

    print("Speedup:", results3[3] / results4[3])

    print("\n")

    #statistics
    filtered = mean_filter_3x3(normalized)
    threshold = 0.65
    results5 = benchmark(compute_image_statistics, filtered, threshold, repeat=10, warmup=1)
    results6 = benchmark(compute_image_statistics_vectorized, filtered, threshold, repeat=10, warmup=1)

    print("Statistics sequential:")
    print_benchmark_report(results5)
    print("\n")

    print("Statistics vectorized:")
    print_benchmark_report(results6)

    print("Speedup:", results5[3] / results6[3])
    print("\n")

    print("Normalization correct:",
      np.allclose(normalize_image(image), normalize_image_vectorized(image)))

    print("Mean filter correct:",
        np.allclose(mean_filter_3x3(normalized),
                    mean_filter_3x3_vectorized(normalized)))
    
    stats_seq = compute_image_statistics(filtered, threshold)
    stats_vec = compute_image_statistics_vectorized(filtered, threshold)

    print("Statistics correct:",
        all(np.isclose(stats_seq[key], stats_vec[key]) for key in stats_seq))
    run_vectorized_pipeline(
    Path("dataset"),
    Path("vectorized_results.csv"),
    0.65
    )
