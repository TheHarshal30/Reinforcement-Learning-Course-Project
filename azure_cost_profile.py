import argparse
import json
import math
import shutil
import subprocess
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AZURE_TRACE_URL = (
    "https://github.com/Azure/AzurePublicDataset/raw/master/data/"
    "AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="azure_trace_artifacts")
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--target-median-cost", type=float, default=3.0)
    parser.add_argument("--min-cost", type=int, default=1)
    parser.add_argument("--clip-max", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def ensure_download(url: str, archive_path: Path, force: bool):
    if archive_path.exists() and not force:
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def ensure_extract(archive_path: Path, extract_dir: Path, force: bool) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(list(extract_dir.glob("*.tsv")) + list(extract_dir.glob("*.txt")))
    if existing and not force:
        return existing[0]
    extractor = shutil.which("unar")
    if extractor:
        subprocess.run(
            [extractor, "-f", "-o", str(extract_dir), str(archive_path)],
            check=True,
        )
    else:
        subprocess.run(
            ["bsdtar", "-xf", str(archive_path), "-C", str(extract_dir)],
            check=True,
        )
    extracted = sorted(list(extract_dir.glob("*.tsv")) + list(extract_dir.glob("*.txt")))
    if not extracted:
        raise FileNotFoundError("No text trace file found after extracting Azure trace archive.")
    return extracted[0]


def reservoir_sample_durations(tsv_path: Path, sample_size: int, seed: int):
    rng = np.random.default_rng(seed)
    sample = np.empty(sample_size, dtype=np.float64)
    seen = 0
    kept = 0
    header_checked = False
    duration_idx = 3
    delimiter = "\t"

    with tsv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            if not header_checked:
                header_checked = True
                if "," in line and "\t" not in line:
                    delimiter = ","
                    parts = line.split(delimiter)
                if "duration" in parts:
                    duration_idx = parts.index("duration")
                    continue
            if len(parts) <= duration_idx:
                parts = line.split()
                if len(parts) <= duration_idx:
                    continue
            try:
                duration = float(parts[duration_idx])
            except ValueError:
                continue
            if duration <= 0.0:
                continue
            seen += 1
            if kept < sample_size:
                sample[kept] = duration
                kept += 1
                continue
            slot = rng.integers(0, seen)
            if slot < sample_size:
                sample[slot] = duration

    return sample[:kept], seen


def lognormal_pdf(x: np.ndarray, mu: float, sigma: float):
    safe_x = np.maximum(x, 1e-12)
    denom = safe_x * sigma * math.sqrt(2.0 * math.pi)
    exponent = -((np.log(safe_x) - mu) ** 2) / (2.0 * sigma * sigma)
    return np.exp(exponent) / denom


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    extract_dir = output_dir / "extracted"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    archive_path = raw_dir / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar"
    ensure_download(AZURE_TRACE_URL, archive_path, args.force_download)
    tsv_path = ensure_extract(archive_path, extract_dir, args.force_extract)

    durations, seen = reservoir_sample_durations(tsv_path, args.sample_size, args.seed)
    if len(durations) == 0:
        raise RuntimeError("No durations sampled from Azure trace.")

    log_durations = np.log(durations)
    mu = float(np.mean(log_durations))
    sigma = float(np.std(log_durations))
    median_duration = float(np.median(durations))
    scale_factor = args.target_median_cost / median_duration
    raw_scaled = np.ceil(durations * scale_factor)
    suggested_clip = int(np.ceil(np.percentile(raw_scaled, 95)))
    clip_max = max(args.min_cost, args.clip_max)
    scaled_costs = np.clip(raw_scaled, args.min_cost, clip_max).astype(int)
    values, counts = np.unique(scaled_costs, return_counts=True)
    probabilities = counts / counts.sum()

    hist_x = np.linspace(max(1e-4, float(np.min(durations))), float(np.percentile(durations, 99.5)), 400)
    hist_pdf = lognormal_pdf(hist_x, mu, max(sigma, 1e-8))
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=80, density=True, alpha=0.65, color="#4C78A8", label="Sampled durations")
    plt.plot(hist_x, hist_pdf, color="#F58518", linewidth=2.0, label="Fitted log-normal PDF")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Density")
    plt.title("Azure Functions Invocation Duration Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "duration_histogram_fit.png", dpi=160)
    plt.close()

    profile = {
        "source_url": AZURE_TRACE_URL,
        "archive_path": str(archive_path),
        "tsv_path": str(tsv_path),
        "rows_seen": int(seen),
        "sample_size_used": int(len(durations)),
        "mu": mu,
        "sigma": sigma,
        "median_duration_seconds": median_duration,
        "mean_duration_seconds": float(np.mean(durations)),
        "p95_duration_seconds": float(np.percentile(durations, 95)),
        "target_median_cost": float(args.target_median_cost),
        "scale_factor": float(scale_factor),
        "clip_max": int(clip_max),
        "suggested_clip_from_p95": int(suggested_clip),
        "azure_cost_values": values.astype(int).tolist(),
        "azure_cost_probabilities": probabilities.astype(float).tolist(),
        "recommended_max_expected_service_cost": float(clip_max),
        "recommended_max_expected_queue_cost": float(72 * clip_max),
    }
    with (output_dir / "azure_cost_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)

    print(output_dir / "azure_cost_profile.json")


if __name__ == "__main__":
    main()
