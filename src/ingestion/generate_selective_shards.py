"""
generate_selective_shards.py
============================
Generates PyTorch-compatible HDF5 shards from isolated UHF PD waveforms,
with precise control over which files from isolated_waveforms/ are included.

Compared to generate_measured_shards.py (which uses ALL available files),
this script lets you:
  - List all available source files and their class assignments.
  - Select exactly which files to include (by filename or class/batch pattern).
  - Assign custom class IDs to each source file.
  - Mix files from different PD types in a single shard collection.

Usage Examples:
  # List all available source files:
  python src/ingestion/generate_selective_shards.py --list

  # Generate shards from specific files only:
  python src/ingestion/generate_selective_shards.py \\
      --include "PD2_Incision_Batch3_14kv.h5" "PD3_Delam_Batch1_7kv.h5" \\
      --num_shards 20 --scenes_per_shard 100

  # Generate with class ID overrides (e.g. treat PD4 and PD5 as same class):
  python src/ingestion/generate_selective_shards.py \\
      --include "PD4_FeOx_Batch1_23kv.h5" "PD5_FeO_High_Batch1_20kv.h5" \\
      --class_ids 3 3 \\
      --num_shards 10

  # Use a glob pattern to include all PD3 files:
  python src/ingestion/generate_selective_shards.py --include "PD3_*"
"""

import argparse
import fnmatch
import logging
import os
import random
import re
import string
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.lineage_tracker import register_root_dataset

INTERIM_ROOT      = PROJECT_ROOT / "data" / "interim_measured" / "isolated_waveforms"
RAW_MEASURED_ROOT = PROJECT_ROOT / "data" / "raw" / "measured"
NOISE_DB_PATH     = PROJECT_ROOT / "data" / "noise" / "real_noise_db.h5"

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

N_SCENE_POINTS = 50_000   # 10 µs at 5 GS/s (dt = 0.2 ns)
NUM_SENSORS    = 4
NICKNAME       = "Selective Measured Shard Generation"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _default_class_id(filename: str) -> int:
    """Infer class ID from filename: PD1=0, PD2=1, ..., PD5=4."""
    m = re.search(r"PD(\d+)", filename, re.IGNORECASE)
    return (int(m.group(1)) - 1) if m else 0


def _scan_available_files() -> list[dict]:
    """Return a list of dicts describing all .h5 files in isolated_waveforms/."""
    available = []
    for h5p in sorted(INTERIM_ROOT.glob("*.h5")):
        size_mb = h5p.stat().st_size / 1e6
        # Count total samples in the file
        n_samples = 0
        try:
            with h5py.File(h5p, "r") as f:
                for voltage in f:
                    n_samples += len(f[voltage])
        except Exception:
            n_samples = -1

        available.append({
            "filename":     h5p.name,
            "path":         h5p,
            "size_mb":      size_mb,
            "n_samples":    n_samples,
            "default_class": _default_class_id(h5p.name),
        })
    return available


def _print_available(available: list[dict]):
    print(f"\n{'Filename':<45} {'Size':>8} {'Samples':>9} {'Default Class'}")
    print("-" * 75)
    for a in available:
        print(f"  {a['filename']:<43} {a['size_mb']:>6.1f}MB {a['n_samples']:>9}  Class {a['default_class']}")
    print(f"\nTotal: {len(available)} file(s) in {INTERIM_ROOT}\n")


def _resolve_includes(include_patterns: list[str], available: list[dict]) -> list[dict]:
    """Match include patterns (exact name or fnmatch glob) against available files."""
    matched = []
    for pattern in include_patterns:
        found = [a for a in available if fnmatch.fnmatch(a["filename"], pattern)]
        if not found:
            log.warning("Pattern '%s' did not match any file.", pattern)
        for f in found:
            if f not in matched:
                matched.append(f)
    return matched


# ──────────────────────────────────────────────────────────────────
# Shard Generation (same core logic as generate_measured_shards.py)
# ──────────────────────────────────────────────────────────────────

def _build_sample_pool(sources: list[dict]) -> list[tuple]:
    """
    Build a flat list of (h5_path, voltage, sample_id, class_id) tuples
    from the resolved source files.
    """
    pool = []
    for src in sources:
        h5p     = src["path"]
        cls_id  = src["class_id"]
        try:
            with h5py.File(h5p, "r") as f:
                for voltage in f:
                    for sample_id in f[voltage]:
                        pool.append((h5p, voltage, sample_id, cls_id))
        except Exception as e:
            log.warning("Could not read %s: %s", h5p.name, e)

    log.info("Sample pool: %d total (voltage, sample) pairs from %d file(s).",
             len(pool), len(sources))
    return pool


def generate_shards(
    sources: list[dict],
    output_dir: Path,
    root_id: str,
    noise_type: str,
    num_shards: int,
    scenes_per_shard: int,
):
    pool = _build_sample_pool(sources)
    if not pool:
        log.error("No samples found in the selected source files. Aborting.")
        sys.exit(1)

    noise_db   = None
    noise_keys = []
    if noise_type == "real":
        if not NOISE_DB_PATH.exists():
            log.error("Real noise requested but %s not found. Run preprocess_noise.py first.", NOISE_DB_PATH)
            sys.exit(1)
        noise_db   = h5py.File(NOISE_DB_PATH, "r")
        noise_keys = list(noise_db["traces"].keys())
        log.info("Loaded real noise database with %d traces.", len(noise_keys))

    output_dir.mkdir(parents=True, exist_ok=True)
    global_pulse_id = 0

    for shard_id in range(1, num_shards + 1):
        batch_scenes = np.zeros((scenes_per_shard, NUM_SENSORS, N_SCENE_POINTS), dtype=np.float64)
        batch_labels = []

        for scene_idx in tqdm(range(scenes_per_shard),
                               desc=f"Shard {shard_id:02d}/{num_shards}", leave=False):
            num_pulses = random.randint(1, 3)

            for _ in range(num_pulses):
                h5p, voltage, sample_id, class_id = random.choice(pool)

                with h5py.File(h5p, "r") as f:
                    grp      = f[voltage][sample_id]
                    channels = list(grp.keys())
                    if not channels:
                        continue

                    signals, starts = {}, {}
                    for ch in channels:
                        ch_data     = grp[ch]
                        signals[ch] = ch_data["iso_signal"][:]
                        starts[ch]  = int(ch_data.attrs["start_idx"])

                # Calculate relative TDOA shifts
                global_s = min(starts.values())
                max_len  = max(len(signals[ch]) + (starts[ch] - global_s) for ch in channels)

                if max_len >= N_SCENE_POINTS - 1000:
                    continue   # Pulse too long for canvas

                idx_in = random.randint(500, N_SCENE_POINTS - max_len - 500)
                global_pulse_id += 1

                for ch in channels:
                    ch_idx    = int(ch.replace("Ch", "")) - 1
                    shift     = starts[ch] - global_s
                    sig       = signals[ch]
                    start_ch  = idx_in + shift
                    end_ch    = start_ch + len(sig)

                    batch_scenes[scene_idx, ch_idx, start_ch:end_ch] += sig

                    bbox_start = max(0, start_ch - 50)
                    bbox_end   = min(N_SCENE_POINTS, end_ch + 50)

                    batch_labels.append([
                        scene_idx, ch_idx, class_id,
                        global_pulse_id - 1, idx_in, bbox_start, bbox_end
                    ])

            # Noise injection
            if noise_type == "synthetic":
                batch_scenes[scene_idx] += np.random.randn(NUM_SENSORS, N_SCENE_POINTS) * 0.05
                t = np.arange(N_SCENE_POINTS) * 0.2e-9
                batch_scenes[scene_idx] += 0.05 * np.sin(2 * np.pi * 100e6 * t)
            elif noise_type == "real" and noise_keys:
                for ch_idx in range(NUM_SENSORS):
                    trace_ds = noise_db["traces"][random.choice(noise_keys)]
                    L = trace_ds.attrs["length"]
                    if L > N_SCENE_POINTS:
                        s = random.randint(0, L - N_SCENE_POINTS - 1)
                        batch_scenes[scene_idx, ch_idx, :] += trace_ds[s:s + N_SCENE_POINTS]

        # Write shard
        out_file = output_dir / f"measured_shard_{shard_id:02d}.h5"
        if out_file.exists():
            out_file.unlink()

        with h5py.File(out_file, "w") as h5f:
            h5f.create_dataset("scenes", data=batch_scenes,
                               dtype=np.float64, compression="gzip", compression_opts=4)
            if batch_labels:
                lbl_arr = np.array(batch_labels, dtype=np.float64).T   # (7, N) — MATLAB format
                h5f.create_dataset("labels", data=lbl_arr,
                                   dtype=np.float64, compression="gzip", compression_opts=4)
                h5f["labels"].attrs["column_1"] = "Scene_ID (0-indexed)"
                h5f["labels"].attrs["column_2"] = "Channel_ID (0-indexed)"
                h5f["labels"].attrs["column_3"] = "Class_ID (0=PD1, 1=PD2, 2=PD3, ...)"
                h5f["labels"].attrs["column_4"] = "Pulse_Instance_ID (0-indexed)"
                h5f["labels"].attrs["column_5"] = "TOA_Index"
                h5f["labels"].attrs["column_6"] = "Start_Idx"
                h5f["labels"].attrs["column_7"] = "End_Idx"

            h5f.attrs["creation_date"]       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h5f.attrs["sampling_frequency_Hz"] = 5e9
            h5f.attrs["time_resolution_s"]   = 0.2e-9
            h5f.attrs["scene_duration_s"]    = 10e-6
            h5f.attrs["num_scenes"]          = scenes_per_shard
            h5f.attrs["num_sensors"]         = NUM_SENSORS
            h5f.attrs["shard_id"]            = shard_id
            h5f.attrs["root_id"]             = root_id
            h5f.attrs["node_id"]             = root_id
            h5f.attrs["origin"]              = "ms"
            h5f.attrs["noise_type"]          = noise_type
            h5f.attrs["source_files"]        = ", ".join(s["path"].name for s in sources)

    if noise_db:
        noise_db.close()

    log.info("Done. Generated %d shards -> %s", num_shards, output_dir)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate UHF measured shards with precise source file selection."
    )
    parser.add_argument("--list", action="store_true",
                        help="List all available source files in isolated_waveforms/ and exit.")
    parser.add_argument("--include", nargs="+", default=None, metavar="PATTERN",
                        help="Filenames or glob patterns to include. E.g. 'PD3_*' 'PD4_FeOx_Batch1_23kv.h5'")
    parser.add_argument("--class_ids", nargs="+", type=int, default=None, metavar="ID",
                        help="Optional: override class ID for each --include pattern (same order). "
                             "If omitted, class IDs are auto-inferred from filenames (PD1=0, PD2=1, ...).")
    parser.add_argument("--noise_type", choices=["none", "synthetic", "real"], default="real",
                        help="Noise type to inject (default: real).")
    parser.add_argument("--num_shards", type=int, default=20,
                        help="Number of output shard files (default: 20).")
    parser.add_argument("--scenes_per_shard", type=int, default=100,
                        help="Scenes per shard (default: 100).")
    args = parser.parse_args()

    available = _scan_available_files()

    # --list mode: just print and exit
    if args.list:
        _print_available(available)
        return

    # Require --include when not just listing
    if not args.include:
        parser.error("Please specify --include PATTERN [PATTERN ...] or use --list to see available files.")

    matched = _resolve_includes(args.include, available)
    if not matched:
        log.error("No source files matched the specified patterns. Use --list to see available files.")
        sys.exit(1)

    # Apply class ID overrides if provided
    if args.class_ids:
        if len(args.class_ids) != len(args.include):
            parser.error(f"--class_ids must have the same number of entries as --include "
                         f"(got {len(args.class_ids)} IDs for {len(args.include)} patterns).")
        # Build a pattern -> class_id map, then re-resolve to apply per matched file
        override_map = {}
        for pattern, cls_id in zip(args.include, args.class_ids):
            for a in available:
                if fnmatch.fnmatch(a["filename"], pattern):
                    override_map[a["filename"]] = cls_id

        for m in matched:
            if m["filename"] in override_map:
                m["class_id"] = override_map[m["filename"]]
            else:
                m["class_id"] = m["default_class"]
    else:
        for m in matched:
            m["class_id"] = m["default_class"]

    # Print summary of what will be used
    print(f"\n{'Selected Source Files':}")
    print("-" * 60)
    for s in matched:
        print(f"  {s['filename']:<45}  -> Class {s['class_id']}")
    print(f"\nTotal: {len(matched)} source file(s)")
    print(f"Shards: {args.num_shards}  |  Scenes/shard: {args.scenes_per_shard}  |  Noise: {args.noise_type}")

    # Lineage
    node_id     = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{run_ts}_ms-{node_id}-{node_id}"
    output_dir  = RAW_MEASURED_ROOT / folder_name

    source_names = ", ".join(s["path"].name for s in matched)
    history_log  = (
        f"Selective measured shard generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"Sources: [{source_names}]. "
        f"Shards: {args.num_shards}, Scenes/shard: {args.scenes_per_shard}, Noise: {args.noise_type}."
    )

    generate_shards(
        sources          = matched,
        output_dir       = output_dir,
        root_id          = node_id,
        noise_type       = args.noise_type,
        num_shards       = args.num_shards,
        scenes_per_shard = args.scenes_per_shard,
    )

    print(f"\nRegistering to lineage database...")
    register_root_dataset(
        origin          = "ms",
        method          = "generate_selective_shards",
        folder_path     = str(output_dir),
        nickname        = NICKNAME,
        history_log     = history_log,
        force_root_id   = node_id,
        force_timestamp = run_ts,
    )
    print(f"[Lineage] Root Dataset Registered: {node_id}")
    print(f"\n[Done] Output -> {output_dir}")


if __name__ == "__main__":
    main()
