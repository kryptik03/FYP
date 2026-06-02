"""
extract_bispectra.py
==========================
Standalone preprocessing script: read raw 1D UHF signal shards, isolate each
individual PD pulse using the labels array, compute a 2D Bispectrum magnitude
grid for each pulse, and save the results as new feature shards.

WHY PER-PULSE (not per-scene)?
-------------------------------
A raw "scene" is a long continuous recording (e.g. 50,001 samples ≈ 10 µs at
5 GHz). Each scene contains multiple sparse PD pulses buried in noise. The
Bispectrum destroys the time axis — unlike the STFT, you cannot compute it
over the whole scene and "slice out" the pulse afterwards.

If you compute the Bispectrum over the entire scene:
  1. Welch's averaging dilutes the PD phase-coupling signature over hundreds
     of noise-only segments, washing it out entirely.
  2. The pad/crop step (signal → 4096 pts) discards everything after the first
     4,096 samples, potentially missing the pulse completely.

The correct approach: use `labels[start_idx : end_idx]` to extract each
isolated pulse waveform FIRST, then compute its Bispectrum.

WHY BISPECTRA?
--------------
The Bispectrum B(f1, f2) = E[X(f1) * X(f2) * X*(f1+f2)] is a third-order
spectral statistic. It captures phase-coupling information between frequency
pairs that is completely invisible to the power spectrum (STFT). This makes
it ideal for distinguishing Partial Discharge (PD) types whose spectral
envelopes overlap but whose phase relationships differ.

STRATEGY — Pulse-level Pad/Crop + Welch Bispectrum
----------------------------------------------------
1.  Read the `labels` array (7, N_pulses) from the raw shard.
    Each column is one (scene, channel, class, pulse_id, toa, start, end) entry.

2.  For each pulse k, slice the raw scene signal:
        raw_pulse = scenes[scene_idx, ch_idx, start_idx : end_idx + 1]
    This gives the isolated PD waveform (typically 200–1000 samples).

3.  Force every isolated pulse to exactly N=`n_fft` points via pad/crop.
    Shorter pulses are zero-padded; longer are truncated.

4.  Compute the 2D Bispectrum magnitude via Welch's segment-averaging method:
        B(f1, f2) = mean_k [ X_k(f1) · X_k(f2) · conj(X_k(f1+f2)) ]
    with nperseg=256, noverlap=128, giving 129 frequency bins per axis.

OUTPUT H5 SCHEMA
----------------
Each output shard mirrors the input but REPLACES `scenes` with:
  pulses_bispectra : float32 array (N_pulses, F, F)
                     where F = nperseg // 2 + 1 = 129
                     Row k corresponds exactly to column k of `labels`.
  labels           : copied verbatim from the input shard
  root attrs       : copied verbatim
  bispectra-specific attrs added:
    bispectrum_padded_len : int (n_fft, e.g. 4096)
    bispectrum_nperseg    : int (256)
    bispectrum_noverlap   : int (128)
    bispectrum_freq_bins  : int (129)
    feature_type          : "bispectra_magnitude_per_pulse"

USAGE
-----
    python src/features/extract_bispectra.py \\
        --input_dir  data/raw/synthesised/... \\
        --parent_node_id 1yGk

    # Optional: override the forced pulse length (default 4096)
    python src/features/extract_bispectra.py \\
        --input_dir  data/raw/cwru/... \\
        --parent_node_id TuKT \\
        --n_fft 2048
"""

import argparse
import glob
import os
import random
import sqlite3
import string
import sys
from datetime import datetime

import h5py
import numpy as np
import scipy.ndimage

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))           # src/features
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))  # FYP/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from src.utils.lineage_tracker import register_process
except ImportError:
    print("[Warning] Could not import lineage_tracker — lineage will not be recorded.")
    register_process = None


# ---------------------------------------------------------------------------
# Label row indices (same schema across all Exp datasets)
# ---------------------------------------------------------------------------
ROW_SCENE_ID   = 0
ROW_CHANNEL_ID = 1
ROW_CLASS_ID   = 2
ROW_PULSE_ID   = 3
ROW_TOA_IDX    = 4
ROW_START_IDX  = 5
ROW_END_IDX    = 6


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def generate_short_id(length: int = 4) -> str:
    """Generate a random 4-character alphanumeric ID for lineage tracking."""
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


# ---------------------------------------------------------------------------
# Core: 1D Pad / Crop
# ---------------------------------------------------------------------------

def pad_or_crop_1d(signal: np.ndarray, n_fft: int) -> np.ndarray:
    """
    *** KEY STEP 1: Uniform Pulse Length ***

    Forces the raw 1D pulse waveform to exactly `n_fft` points so that every
    bispectrum has identical frequency-bin spacing and alignment.

    - If len(signal) < n_fft  → zero-pad at the END (causal padding).
    - If len(signal) > n_fft  → keep only the FIRST n_fft samples (truncate).

    Args:
        signal : 1D numpy array of arbitrary length (the isolated pulse).
        n_fft  : Target length (default 4096).

    Returns:
        Fixed-length 1D numpy array of shape (n_fft,), dtype float32.
    """
    signal = signal.astype(np.float32)
    L = len(signal)

    if L == n_fft:
        return signal                                       # Already correct

    if L < n_fft:
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[:L] = signal
        return padded

    # L > n_fft → truncate
    return signal[:n_fft]


# ---------------------------------------------------------------------------
# Core: 2D Bispectrum Computation (Welch segment-averaging method)
# ---------------------------------------------------------------------------

# Welch segmentation parameters (fixed to give 129 frequency bins per axis)
_NPERSEG  = 256
_NOVERLAP = 128

def compute_bispectrum_welch(signal: np.ndarray, nperseg: int = 256, noverlap: int = 128, L: int = 1) -> np.ndarray:
    """
    Computes the Bispectrum magnitude using the exact equations from the paper:
    - Zero-mean assumption
    - Segment-averaging (Eq 8)
    - Frequency-domain smoothing (Eq 7) using window parameter L.
    
    Args:
        signal   : 1D float32 array of fixed length (n_fft points).
        nperseg  : Welch segment length (default 256).
        noverlap : Overlap between consecutive segments (default 128).
        L        : Smoothing window radius for Eq 7 (default 1 yields a 3x3 filter).
    """
    step   = nperseg - noverlap
    starts = np.arange(0, len(signal) - nperseg + 1, step)

    n_bins = nperseg // 2 + 1    # 129
    bispectrum_avg = np.zeros((n_bins, n_bins), dtype=np.complex64)

    f1_idx  = np.arange(n_bins, dtype=np.int32)
    f2_idx  = np.arange(n_bins, dtype=np.int32)
    f12_idx = np.clip(f1_idx[:, None] + f2_idx[None, :], 0, n_bins - 1)

    for start in starts:
        segment = signal[start : start + nperseg]
        
        # PAPER FIX 1: Enforce zero-mean assumption
        segment = segment - np.mean(segment)
        
        segment = segment * np.hanning(nperseg)
        X       = np.fft.rfft(segment, n=nperseg)
        
        # Raw complex bispectrum for this segment
        raw_b = X[f1_idx[:, None]] * X[f2_idx[None, :]] * np.conj(X[f12_idx])
        bispectrum_avg += raw_b

    # Equation 8: Average across segments
    bispectrum_avg /= max(len(starts), 1)
    
    # PAPER FIX 2: Equation 7 (Frequency-domain smoothing)
    # A moving average of size (2L+1, 2L+1) applied to the complex bispectrum
    if L > 0:
        window_size = 2 * L + 1
        # Smooth real and imaginary components separately
        real_smoothed = scipy.ndimage.uniform_filter(bispectrum_avg.real, size=window_size)
        imag_smoothed = scipy.ndimage.uniform_filter(bispectrum_avg.imag, size=window_size)
        bispectrum_avg = real_smoothed + 1j * imag_smoothed

    # Extract magnitude for the neural network
    return np.abs(bispectrum_avg).astype(np.float32)


# ---------------------------------------------------------------------------
# Shard Processing  ← CORE FIX: iterate over pulses, not scenes
# ---------------------------------------------------------------------------

def process_shard(input_path: str, output_path: str, n_fft: int):
    """
    Read one raw 1D shard, compute a Bispectrum for every individual PD pulse
    (indexed by the labels array), and write the result to a new output shard.

    H5 Input Schema (raw shards):
        scenes      : float32  (N_scenes, N_channels, T_samples)
        labels      : float/int (7, N_pulses)  — row indices defined at top of file
        attrs       : sampling_frequency_Hz, time_resolution_s, ...

    H5 Output Schema (bispectrum feature shards):
        pulses_bispectra : float32 (N_pulses, F, F)
                           Row k = bispectrum of the pulse described by labels[:, k].
                           F = _NPERSEG // 2 + 1 = 129.
        labels           : copied verbatim (column k still matches row k of bispectra)
        attrs            : all original attrs + bispectrum metadata
    """
    n_bins = _NPERSEG // 2 + 1   # 129

    with h5py.File(input_path, "r") as h5_in:
        if "scenes" not in h5_in or "labels" not in h5_in:
            print(f"    [Skip] Missing 'scenes' or 'labels' dataset in {input_path}")
            return

        scenes_1d = h5_in["scenes"][:]   # (N_scenes, N_channels, T_samples)
        labels    = h5_in["labels"][:]   # (7, N_pulses)

        N_scenes, N_channels, T_samples = scenes_1d.shape
        N_pulses = labels.shape[1]

        print(f"    Input:  {N_scenes} scenes × {N_channels} ch × {T_samples} samples, "
              f"{N_pulses} pulses")

        # Allocate output: one bispectrum per pulse entry in the labels table
        pulses_bispectra = np.zeros(
            (N_pulses, n_bins, n_bins), dtype=np.float32
        )

        for k in range(N_pulses):
            scene_idx = int(labels[ROW_SCENE_ID,   k])
            ch_idx    = int(labels[ROW_CHANNEL_ID, k])
            start_idx = int(labels[ROW_START_IDX,  k])
            end_idx   = int(labels[ROW_END_IDX,    k])

            # *** KEY FIX: Slice out the isolated PD pulse waveform ***
            raw_pulse = scenes_1d[scene_idx, ch_idx, start_idx : end_idx + 1]

            # Uniform length via pad/crop (ensures consistent Welch segment count)
            uniform_pulse = pad_or_crop_1d(raw_pulse, n_fft)

            # Compute 2D bispectrum of the isolated pulse
            pulses_bispectra[k] = compute_bispectrum_welch(
                uniform_pulse, nperseg=_NPERSEG, noverlap=_NOVERLAP
            )

        # Read labels dataset for copying (with its attrs)
        labels_data  = labels
        labels_attrs = dict(h5_in["labels"].attrs)
        root_attrs   = dict(h5_in.attrs)

    # Write output shard
    with h5py.File(output_path, "w") as h5_out:
        # *** KEY OUTPUT: pulses_bispectra — one row per label entry ***
        h5_out.create_dataset(
            "pulses_bispectra",
            data=pulses_bispectra,
            compression="gzip",
            compression_opts=4,
        )

        ds = h5_out.create_dataset("labels", data=labels_data)
        for k, v in labels_attrs.items():
            ds.attrs[k] = v

        # Copy all original root attributes
        for k, v in root_attrs.items():
            h5_out.attrs[k] = v

        # Add bispectrum-specific metadata
        h5_out.attrs["bispectrum_padded_len"]  = n_fft
        h5_out.attrs["bispectrum_nperseg"]     = _NPERSEG
        h5_out.attrs["bispectrum_noverlap"]    = _NOVERLAP
        h5_out.attrs["bispectrum_freq_bins"]   = n_bins
        h5_out.attrs["bispectrum_L"]           = 1
        h5_out.attrs["feature_type"]           = "bispectra_magnitude_per_pulse"

    out_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"    Output: {output_path}  ({out_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_features(
    input_dir: str,
    output_root: str,
    parent_node_id: str,
    n_fft: int = 4096,
):
    """
    Process every shard in `input_dir` and write per-pulse bispectra to a new
    timestamped folder under `output_root/bispectra/`.
    Also registers the run in the SQLite lineage database.
    """
    node_id   = generate_short_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Query lineage DB for parent metadata (origin tag + root_id)
    db_path = os.path.abspath(
        os.path.join(_SCRIPT_DIR, "..", "utils", "lineage.db")
    )
    origin, root_id = "unk", "unk"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT origin, root_id FROM nodes WHERE node_id=?", (parent_node_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            origin, root_id = row

    # Build output folder: data/features/bispectra/<timestamp>-<origin>-<root_id>-<node_id>/
    folder_name       = f"{timestamp}-{origin}-{root_id}-{node_id}"
    target_output_dir = os.path.join(output_root, "bispectra", folder_name)
    os.makedirs(target_output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    if not h5_files:
        print(f"[Error] No .h5 files found in: {input_dir}")
        return

    print("=" * 60)
    print(f"  Bispectrum Feature Extraction  (PER-PULSE)")
    print(f"  Input    : {input_dir}")
    print(f"  Output   : {target_output_dir}")
    print(f"  n_fft    : {n_fft}  (pad/crop target per pulse)")
    print(f"  nperseg  : {_NPERSEG}  →  F bins = {_NPERSEG // 2 + 1}")
    print(f"  Shards   : {len(h5_files)}")
    print("=" * 60)

    for h5_file in h5_files:
        fname = os.path.basename(h5_file)
        out_f = os.path.join(target_output_dir, fname)
        print(f"\n[Shard] {fname}")
        process_shard(h5_file, out_f, n_fft)

    history_log = (
        f"Per-pulse bispectrum extraction. "
        f"n_fft={n_fft} (pad/crop per pulse), nperseg={_NPERSEG}, "
        f"noverlap={_NOVERLAP}, freq_bins={_NPERSEG // 2 + 1}. "
        f"Shards={len(h5_files)}."
    )

    if register_process is not None:
        register_process(
            parent_id        = parent_node_id,
            stage            = "feature_extraction",
            method           = "bispectrum_per_pulse",
            folder_path      = target_output_dir,
            appended_history = history_log,
            force_node_id    = node_id,
            force_timestamp  = timestamp,
        )
        print(f"\n[Lineage] Registered node {node_id} (child of {parent_node_id}).")

    print(f"\n[Done] Per-pulse bispectra saved to: {target_output_dir}")
    return target_output_dir


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-pulse 2D Bispectrum magnitude features from raw 1D H5 shards."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing raw .h5 shards (must have 'scenes' and 'labels' datasets)."
    )
    parser.add_argument(
        "--output_root", type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "features")
        ),
        help="Root directory for feature outputs. Subdirectory 'bispectra/' will be created."
    )
    parser.add_argument(
        "--parent_node_id", type=str, required=True,
        help="Lineage node ID of the raw dataset (used for DAG registration)."
    )
    parser.add_argument(
        "--n_fft", type=int, default=4096,
        help="Target length for pad/crop of each isolated pulse before Welch FFT (default: 4096)."
    )

    args = parser.parse_args()
    extract_features(
        input_dir      = os.path.abspath(args.input_dir),
        output_root    = os.path.abspath(args.output_root),
        parent_node_id = args.parent_node_id,
        n_fft          = args.n_fft,
    )
