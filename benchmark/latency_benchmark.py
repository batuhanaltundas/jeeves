"""
Jeeves Language Model Refinement Benchmark

Benchmarks multiple local LLMs as speech-transcript
refinement models.

Variables:
    - model
    - transcript length

Measured:
    - model loading time
    - refinement latency
    - output token count
    - input token count

Outputs:
    evaluation/results/model_latency_results.csv
    evaluation/results/model_latency_summary.csv
    evaluation/results/model_latency.png
"""

from __future__ import annotations

import csv
import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

# add path to the llm directory
import os
import sys
# Get the absolute path of the current file's parent directory
current_dir = Path(__file__).resolve().parent

# Get the grand-parent directory (the parent of the current directory)
parent_dir = current_dir.parent

# Add it to the system path
sys.path.append(str(parent_dir))

from llm.refiner import TextRefiner


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path(
    "evaluation/results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Number of repetitions per condition.
REPETITIONS = 3


# Transcript lengths.
SCRIPT_LENGTHS = [
    5,
    10,
    20,
    40,
    80,
    120,
]


# ============================================================
# Models
# ============================================================

MODELS = {
    "Qwen2.5-0.5B": (
        "Qwen/Qwen2.5-0.5B-Instruct"
    ),

    "Qwen2.5-1.5B": (
        "Qwen/Qwen2.5-1.5B-Instruct"
    ),

    "SmolLM2-1.7B": (
        "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    ),

    # Uncomment if desired.
    #
    # "Llama-3.2-1B": (
    #     "meta-llama/Llama-3.2-1B-Instruct"
    # ),
}


# ============================================================
# Test scripts
# ============================================================

BASE_SENTENCES = [

    "Move the Unit to the left.",

    "Move Unit two behind the building.",

    (
        "Send the second Unit north and keep "
        "the first Unit here."
    ),

    (
        "Move Unit two toward the northern "
        "objective and maintain formation."
    ),

    (
        "Move the second Unit behind the building, "
        "then send the first Unit toward the "
        "western objective."
    ),

    (
        "Move Unit two toward the northern objective "
        "while Unit one maintains its current position. "
        "After Unit two reaches the building, move "
        "Unit one to the western side."
    ),

    (
        "Move the second Unit toward the northern "
        "objective while keeping the first Unit behind "
        "the central building. Once the second Unit "
        "reaches the objective, reposition the first "
        "Unit toward the western approach. Do not "
        "expose either unit to unnecessary enemy fire "
        "and preserve the current formation until the "
        "next command."
    ),
]


# ============================================================
# Data
# ============================================================

@dataclass
class BenchmarkResult:

    model: str

    model_id: str

    target_words: int

    actual_words: int

    repetition: int

    latency_seconds: float

    output_words: int


# ============================================================
# Utilities
# ============================================================

def word_count(
    text: str,
) -> int:

    return len(
        text.split()
    )


def build_script(
    target_words: int,
) -> str:

    """
    Construct an approximately target-length
    transcript.
    """

    source = BASE_SENTENCES[
        min(
            len(BASE_SENTENCES) - 1,
            max(
                0,
                target_words // 20,
            ),
        )
    ]

    words = source.split()

    if len(words) >= target_words:

        return " ".join(
            words[:target_words]
        )

    result = []

    index = 0

    while len(result) < target_words:

        result.append(
            words[
                index % len(words)
            ]
        )

        index += 1

    return " ".join(
        result[:target_words]
    )


# ============================================================
# Synchronization
# ============================================================

def synchronize():

    if torch.cuda.is_available():

        torch.cuda.synchronize()


# ============================================================
# Benchmark one model
# ============================================================

def benchmark_model(
    model_name: str,
    model_id: str,
):

    print()
    print("=" * 70)
    print(
        f"MODEL: {model_name}"
    )
    print(
        f"ID: {model_id}"
    )
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    print(
        "Loading model..."
    )

    synchronize()

    load_start = time.perf_counter()

    refiner = TextRefiner(
        model_name=model_id
    )

    synchronize()

    load_end = time.perf_counter()

    load_time = (
        load_end
        - load_start
    )

    print(
        f"Model load time: "
        f"{load_time:.3f}s"
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        "Running warmup..."
    )

    warmup_text = (
        "Move Unit two toward "
        "the northern objective."
    )

    refiner.refine(
        warmup_text
    )

    refiner.refine(
        warmup_text
    )

    print(
        "Warmup complete."
    )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    results = []

    for target_words in SCRIPT_LENGTHS:

        script = build_script(
            target_words
        )

        actual_words = word_count(
            script
        )

        print(
            f"\n{actual_words} words:"
        )

        for repetition in range(
            REPETITIONS
        ):

            synchronize()

            start = time.perf_counter()

            output = refiner.refine(
                script
            )

            synchronize()

            end = time.perf_counter()

            latency = (
                end - start
            )

            result = BenchmarkResult(
                model=model_name,
                model_id=model_id,
                target_words=target_words,
                actual_words=actual_words,
                repetition=repetition + 1,
                latency_seconds=latency,
                output_words=word_count(
                    output
                ),
            )

            results.append(
                result
            )

            print(
                f"  Run {repetition + 1}: "
                f"{latency:.3f}s"
            )

    # --------------------------------------------------------
    # Free model
    # --------------------------------------------------------

    del refiner

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print(
        "\nModel unloaded."
    )

    return results, load_time


# ============================================================
# Main benchmark
# ============================================================

def benchmark():

    all_results = []

    load_times = {}

    for model_name, model_id in MODELS.items():

        results, load_time = (
            benchmark_model(
                model_name,
                model_id,
            )
        )

        all_results.extend(
            results
        )

        load_times[
            model_name
        ] = load_time

    # --------------------------------------------------------
    # Raw CSV
    # --------------------------------------------------------

    raw_file = (
        RESULTS_DIR
        / "model_latency_results.csv"
    )

    with open(
        raw_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "model",
            "model_id",
            "target_words",
            "actual_words",
            "repetition",
            "latency_seconds",
            "output_words",
        ])

        for result in all_results:

            writer.writerow([
                result.model,
                result.model_id,
                result.target_words,
                result.actual_words,
                result.repetition,
                result.latency_seconds,
                result.output_words,
            ])

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = []

    for model_name in MODELS:

        for length in SCRIPT_LENGTHS:

            samples = [
                r.latency_seconds
                for r in all_results
                if (
                    r.model == model_name
                    and
                    r.target_words == length
                )
            ]

            summary.append({
                "model": model_name,
                "words": length,
                "mean": statistics.mean(
                    samples
                ),
                "median": statistics.median(
                    samples
                ),
                "std": (
                    statistics.stdev(
                        samples
                    )
                    if len(samples) > 1
                    else 0
                ),
                "min": min(samples),
                "max": max(samples),
            })

    summary_file = (
        RESULTS_DIR
        / "model_latency_summary.csv"
    )

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "model",
            "script_length_words",
            "mean_latency_seconds",
            "median_latency_seconds",
            "std_latency_seconds",
            "min_latency_seconds",
            "max_latency_seconds",
        ])

        for row in summary:

            writer.writerow([
                row["model"],
                row["words"],
                row["mean"],
                row["median"],
                row["std"],
                row["min"],
                row["max"],
            ])

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name in MODELS:

        print()
        print(
            model_name
        )

        for length in SCRIPT_LENGTHS:

            row = next(
                x
                for x in summary
                if (
                    x["model"] == model_name
                    and
                    x["words"] == length
                )
            )

            print(
                f"{length:>4} words | "
                f"{row['mean']:.3f}s ± "
                f"{row['std']:.3f}s"
            )

        print(
            f"Load time: "
            f"{load_times[model_name]:.3f}s"
        )

    # --------------------------------------------------------
    # Generate figure
    # --------------------------------------------------------

    generate_figure(
        summary
    )

    print()
    print(
        f"Raw results: {raw_file}"
    )

    print(
        f"Summary: {summary_file}"
    )


# ============================================================
# Figure
# ============================================================

def generate_figure(
    summary,
):

    plt.figure(
        figsize=(9, 6)
    )

    for model_name in MODELS:

        rows = [
            x
            for x in summary
            if x["model"] == model_name
        ]

        lengths = [x["words"] for x in rows]
        means = np.array([x["mean"] for x in rows])
        stds = np.array([x["std"] for x in rows])

        plt.plot(
            lengths,
            means,
            label=model_name,
        )
        
        plt.fill_between(
            lengths,
            means - stds,
            means + stds,
            alpha=0.2,
        )
    plt.xlabel(
        "Transcript Length (words)"
    )

    plt.ylabel(
        "Refinement Latency (seconds)"
    )

    plt.title(
        "Local LLM Refinement Latency "
        "vs. Transcript Length"
    )
    
    # 2. Position the legend below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncols=2)

    # 3. Prevent the legend from getting cut off
    plt.tight_layout()
    plt.tight_layout()

    output = (
        RESULTS_DIR
        / "model_latency.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nFigure saved to: {output}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    benchmark()