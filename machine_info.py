import platform
import psutil
import time 
import statistics
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from sequential_pipeline import run_sequential_pipeline
import cProfile
import pstats


def machine_info():
    mem = psutil.virtual_memory()
    machine_card={
        "architecture":platform.machine(),
        "operating_system":platform.system(),
        "processor_name":platform.processor(),
        "physical_core":psutil.cpu_count(logical=False),
        "logical_cpu":psutil.cpu_count(logical=True),
        "logical_cpus_per_core": psutil.cpu_count(logical=True) / psutil.cpu_count(logical=False),
        "CPU_freq":psutil.cpu_freq(),
        "total RAM":mem.total,
        "current_RAM":mem.available,
        "utilization_of_CPU":psutil.cpu_percent(interval=1, percpu=True)
    }
    return machine_card
      

def print_machine_card(info):
    for cle,valeur in info.items():
        print(f"{cle} : {valeur}")


def display_cpu_bars():
    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)

    for i in range(len(cpu_usage)):
        print("CPU", i, ":", cpu_usage[i], "%")
    

def measure_perf_counter(function, *args): #le nb de fois que l'horloge interne du CPU varie 
    t1=time.perf_counter()
    function(*args)
    t2=time.perf_counter()
    print(t2-t1)

def benchmark(function, *args, repeat=10, warmup=0):
    for i in range(warmup):
        function(*args)
    tab = [0] * repeat
    for i in range(repeat):
        t1 = time.perf_counter()
        function(*args)
        t2 = time.perf_counter()
        tab[i] = t2 - t1
    min_time = min(tab)
    max_time = max(tab)
    mean = statistics.mean(tab)
    median = statistics.median(tab)
    std = statistics.stdev(tab)
    cv_percent = (std / mean) * 100
    return tab, min_time, max_time, mean, median, std, cv_percent

def print_benchmark_report(results):
    times, min_time, max_time, mean, median, std, cv_percent = results
    print("Times :", times)
    print("Min :", min_time)
    print("Max :", max_time)
    print("Mean :", mean)
    print("Median :", median)
    print("Standard deviation :", std)
    print("CV (%) :", cv_percent)

def save_benchmark_csv(results, filename):
    times, min_time, max_time, mean, median, std, cv_percent=results
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([f"time_{i+1}" for i in range(len(times))]+
            ["min", "max", "mean", "median", "std", "cv_percent"])
        writer.writerow(times+[min_time, max_time, mean, median, std, cv_percent])

def visualisation(results):
    RESULTS = Path("results")
    RESULTS.mkdir(exist_ok=True)

    save_benchmark_csv(results, RESULTS / "csv_benchmark.csv")

    with open(RESULTS / "csv_benchmark.csv", "r", encoding="utf-8") as fichier_csv:
        reader = csv.DictReader(fichier_csv, delimiter=",")
        row = next(reader)

    times = []
    for key in row:
        if key.startswith("time_"):
            times.append(float(row[key]))

    mean = float(row["mean"])
    median = float(row["median"])

    x = range(1, len(times) + 1)

    plt.plot(x, times, marker="o", label="Timing")
    plt.axhline(mean, linestyle="--", label="mean")
    plt.axhline(median, linestyle="--", label="median")

    plt.xlabel("Run")
    plt.ylabel("Time(s)")
    plt.title("Run Variability")
    plt.legend()

    plt.savefig(RESULTS / "run_variability.png")
    plt.close()
    # Graph 2: Runtime Histogram
    plt.hist(times, bins=5)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.title("Runtime Histogram")
    plt.savefig(RESULTS / "runtime_histogram.png")
    plt.close()

    # Graph 3: Runtime Boxplot
    plt.boxplot(times)
    plt.ylabel("Time (s)")
    plt.title("Runtime Boxplot")
    plt.savefig(RESULTS / "runtime_boxplot.png")
    plt.close()

def profiling():
    RESULTS = Path("results")
    RESULTS.mkdir(exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()

    run_sequential_pipeline(
        Path("dataset"),
        Path("sequential_results.csv"),
        0.65
    )

    profiler.disable()
    profiler.dump_stats(str(RESULTS / "image_pipeline.prof"))

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(10)

if __name__ == "__main__":
    info=machine_info()
    print_machine_card(info)
    display_cpu_bars()
    results = benchmark(
         run_sequential_pipeline,
         Path("dataset"),
         Path("sequential_results.csv"),
         0.65,
         repeat=10,
         warmup=1
    )

    print_benchmark_report(results)
    save_benchmark_csv(results, "baseline_results.csv")
    visualisation(results)
    profiling()