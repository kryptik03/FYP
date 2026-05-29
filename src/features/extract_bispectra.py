"""
extract_bispectra.py
==========================
Standalone preprocessing script: read raw 1D UHF signal shards and compute
2D Bispectrum magnitude grids, saving them as new feature shards.

WHY BISPECTRA?
--------------
The Bispectrum B(f1, f2) = E[X(f1) * X(f2) * X*(f1+f2)] is a third-order
spectral statistic. It captures phase-coupling information between frequency
pairs that is completely invisible to the power spectrum (STFT). This makes
it ideal for distinguishing Partial Discharge (PD) types whose spectral
envelopes overlap but whose phase relationships differ.

STRATEGY — "Pad/Crop + Extract Positive Quadrant"
--------------------------------------------------
1.  Force every raw 1D signal to exactly N=4096 points.
    - Shorter signals are ZERO-PADDED at the end.
    - Longer signals are TRUNCATED (first 4096 points kept).
    This ensures strict frequency-bin alignment across all 4 datasets.

2.  Compute the 2D Bispectrum magnitude via direct FFT:
    B(f1, f2) = X(f1) * X(f2) * conj(X(f1+f2))
    The full grid is (N//2+1) x (N//2+1) = 2049 x 2049.

3.  Extract only the Non-Redundant Triangle (NRT) — the positive
    first-quadrant region where f1 >= 0 and f2 >= 0. The full bispectrum
    has 8-fold symmetry; the NRT alone is sufficient.
    Output shape per pulse: (N//2+1, N//2+1) = (2049, 2049) as float32.

OUTPUT H5 SCHEMA
----------------
Each output shard mirrors the input but replaces `scenes` with:
  scenes_bispectra : float32 array (N_scenes, N_channels, F, F)
                     where F = n_fft // 2 + 1 (e.g. 2049 for N=4096)
  labels           : copied verbatim from the input shard
  root attrs       : copied verbatim
  bispectra-specific attrs added:
    bispectrum_n_fft    : int (4096)
    bispectrum_freq_bins: int (2049)
    feature_type        : "bispectra_magnitude"

USAGE
-----
    python src/features/extract_bispectra.py \\
        --input_dir  data/raw/cwru/20260524_233633_cw-TuKT-TuKT \\
        --output_root data/features \\
        --parent_node_id TuKT

    # Optional: override the forced signal length (default 4096)
    python src/features/extract_bispectra.py \\
        --input_dir  data/raw/synthesised/... \\
        --output_root data/features \\
        --parent_node_id N5ZR \\
        --n_fft 4096
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
    *** KEY STEP 1: Uniform Signal Length ***

    Forces the raw 1D signal to exactly `n_fft` points so that every bispectrum
    has identical frequency-bin spacing and alignment.

    - If len(signal) < n_fft  → zero-pad at the END (causal padding).
    - If len(signal) > n_fft  → keep only the FIRST n_fft samples (truncate).

    Args:
        signal : 1D numpy array of arbitrary length.
        n_fft  : Target length (default 4096).

    Returns:
        Fixed-length 1D numpy array of shape (n_fft,), dtype float32.
    """
    signal = signal.astype(np.float32)
    L = len(signal)

    if L == n_fft:
        return signal                                       # Already correct — nothing to do

    if L < n_fft:
        # Zero-pad at the end  ← "PAD" branch
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[:L] = signal
        return padded

    # L > n_fft  → truncate  ← "CROP" branch
    return signal[:n_fft]


# ---------------------------------------------------------------------------
# Core: 2D Bispectrum Computation
# ---------------------------------------------------------------------------

def compute_bispectrum(signal: np.ndarray, n_fft: int) -> np.ndarray:
    """
    Compute the 2D Bispectrum magnitude for a single 1D signal.

    *** KEY STEP 2: Bispectrum Computation on Uniform N-point Signal ***

    The bispectrum is defined as:
        B(f1, f2) = X(f1) · X(f2) · X*(f1 + f2)

    where X(f) = FFT(signal)[f], the single-sided complex spectrum.

    IMPLEMENTATION NOTES
    --------------------
    - We use the single-sided (positive-frequency) DFT of length N=4096.
    - The resulting full bispectrum grid has shape (N//2+1, N//2+1).
    - We extract the positive-frequency "Non-Redundant Triangle" quadrant:
        f1 in [0, N//2],  f2 in [0, N//2].
    - The f1+f2 index is clamped / wrapped mod N (to stay within the
      single-sided spectrum via the symmetry X(N-k) = X*(k) for real signals).

    Args:
        signal : 1D numpy array of exactly n_fft points (already padded/cropped).
        n_fft  : FFT length (must equal len(signal), typically 4096).

    Returns:
        bispectrum : 2D float32 array of shape (n_fft//2+1, n_fft//2+1)
                     containing the bispectrum magnitude |B(f1, f2)|.
                     This is the positive-frequency quadrant only (saves space
                     and is sufficient due to 8-fold symmetry).
    """
    n_bins = n_fft // 2 + 1   # e.g. 2049 for n_fft=4096

    # Full complex FFT, then keep only single-sided spectrum [0 .. N/2]
    X_full = np.fft.rfft(signal, n=n_fft)  # shape: (n_bins,), dtype complex64/128

    # *** KEY STEP 3: Extract Positive Frequency Quadrant ***
    # We loop over f1, f2 in [0, n_bins) and compute |X(f1) * X(f2) * X*(f1+f2)|.
    # Vectorised using broadcasting for speed:
    #   - f1 indices as column vector (n_bins, 1)
    #   - f2 indices as row vector    (1, n_bins)
    #   - f1+f2 index clamped to [0, n_bins-1]  (beyond Nyquist: real-signal symmetry)

    f1_idx = np.arange(n_bins, dtype=np.int32)   # shape (n_bins,)
    f2_idx = np.arange(n_bins, dtype=np.int32)   # shape (n_bins,)

    # f1+f2 grid; clamp at n_bins-1 (Nyquist ceiling for single-sided spectrum)
    # For a real signal: X(N-k) = X*(k), so frequencies > Nyquist fold back.
    f12_idx = np.clip(
        f1_idx[:, None] + f2_idx[None, :],   # (n_bins, n_bins)
        0, n_bins - 1
    )

    # Vectorised bispectrum magnitude
    #   |B(f1,f2)| = |X(f1)| * |X(f2)| * |X*(f1+f2)|
    #              = |X(f1)| * |X(f2)| * |X(f1+f2)|   (magnitude symmetric)
    bispectrum = (
        np.abs(X_full[f1_idx[:, None]])    # (n_bins, 1)  → broadcast to (n_bins, n_bins)
        * np.abs(X_full[f2_idx[None, :]])  # (1, n_bins)
        * np.abs(X_full[f12_idx])          # (n_bins, n_bins)
    ).astype(np.float32)

    return bispectrum   # shape: (n_bins, n_bins) = (2049, 2049) for n_fft=4096


# ---------------------------------------------------------------------------
# Shard Processing
# ---------------------------------------------------------------------------

def process_shard(input_path: str, output_path: str, n_fft: int):
    """
    Read one raw 1D shard, compute bispectra for every (scene, channel) pair,
    and write the result to a new output shard.

    H5 Input Schema (raw shards):
        scenes      : float32 (N_scenes, N_channels, T_samples)   ← variable T_samples
        labels      : int/float (7, N_pulses)
        attrs       : sampling_frequency_Hz, time_resolution_s, ...

    H5 Output Schema (bispectrum feature shards):
        scenes_bispectra : float32 (N_scenes, N_channels, F, F)
                           where F = n_fft // 2 + 1  (e.g. 2049)
        labels           : copied verbatim
        attrs            : all original attrs + bispectrum metadata
    """
    n_bins = n_fft // 2 + 1   # e.g. 2049

    with h5py.File(input_path, "r") as h5_in:
        scenes_1d = h5_in["scenes"][:]   # (N_scenes, N_channels, T_samples)
        N_scenes, N_channels, T_samples = scenes_1d.shape

        print(f"    Input: {N_scenes} scenes × {N_channels} ch × {T_samples} samples")

        # Allocate output array upfront (much faster than incremental append)
        # Shape: (N_scenes, N_channels, n_bins, n_bins)
        bispectra = np.zeros(
            (N_scenes, N_channels, n_bins, n_bins), dtype=np.float32
        )

        for s_idx in range(N_scenes):
            for c_idx in range(N_channels):
                raw_sig = scenes_1d[s_idx, c_idx, :]    # shape: (T_samples,)

                # *** STEP 1: Pad or crop to exactly n_fft points ***
                uniform_sig = pad_or_crop_1d(raw_sig, n_fft)

                # *** STEP 2: Compute 2D bispectrum magnitude ***
                bispectrum_grid = compute_bispectrum(uniform_sig, n_fft)

                bispectra[s_idx, c_idx] = bispectrum_grid   # (n_bins, n_bins)

        # Copy labels and root attributes
        labels_data  = h5_in["labels"][:] if "labels" in h5_in else None
        labels_attrs = dict(h5_in["labels"].attrs) if "labels" in h5_in else {}
        root_attrs   = dict(h5_in.attrs)

    # Write output shard
    with h5py.File(output_path, "w") as h5_out:
        # *** KEY OUTPUT: scenes_bispectra ***
        h5_out.create_dataset(
            "scenes_bispectra",
            data=bispectra,
            compression="gzip",   # ~3-5× size reduction for smooth surfaces
            compression_opts=4,
        )

        if labels_data is not None:
            ds = h5_out.create_dataset("labels", data=labels_data)
            for k, v in labels_attrs.items():
                ds.attrs[k] = v

        # Copy all original root attributes (preserves time_resolution_s etc.)
        for k, v in root_attrs.items():
            h5_out.attrs[k] = v

        # Add bispectrum-specific metadata
        h5_out.attrs["bispectrum_n_fft"]     = n_fft
        h5_out.attrs["bispectrum_freq_bins"] = n_bins
        h5_out.attrs["feature_type"]         = "bispectra_magnitude"

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
    Process every shard in `input_dir` and write bispectra to a new
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
    folder_name      = f"{timestamp}-{origin}-{root_id}-{node_id}"
    target_output_dir = os.path.join(output_root, "bispectra", folder_name)
    os.makedirs(target_output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    if not h5_files:
        print(f"[Error] No .h5 files found in: {input_dir}")
        return

    print("=" * 60)
    print(f"  Bispectrum Feature Extraction")
    print(f"  Input    : {input_dir}")
    print(f"  Output   : {target_output_dir}")
    print(f"  n_fft    : {n_fft}  (pad/crop target)")
    print(f"  F bins   : {n_fft // 2 + 1}  (per axis)")
    print(f"  Shards   : {len(h5_files)}")
    print("=" * 60)

    for h5_file in h5_files:
        fname   = os.path.basename(h5_file)
        out_f   = os.path.join(target_output_dir, fname)
        print(f"\n[Shard] {fname}")
        process_shard(h5_file, out_f, n_fft)

    history_log = (
        f"Bispectrum extraction. "
        f"n_fft={n_fft}, freq_bins={n_fft // 2 + 1}. "
        f"Pad/crop strategy. "
        f"Shards={len(h5_files)}."
    )

    if register_process is not None:
        register_process(
            parent_id        = parent_node_id,
            stage            = "feature_extraction",
            method           = "bispectrum",
            folder_path      = target_output_dir,
            appended_history = history_log,
            force_node_id    = node_id,
            force_timestamp  = timestamp,
        )
        print(f"\n[Lineage] Registered node {node_id} (child of {parent_node_id}).")

    print(f"\n[Done] Bispectra saved to: {target_output_dir}")
    return target_output_dir


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract 2D Bispectrum magnitude features from raw 1D H5 shards."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing raw .h5 shards (must have a 'scenes' dataset)."
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
        help="Target signal length for pad/crop before FFT (default: 4096)."
    )

    args = parser.parse_args()
    extract_features(
        input_dir      = os.path.abspath(args.input_dir),
        output_root    = os.path.abspath(args.output_root),
        parent_node_id = args.parent_node_id,
        n_fft          = args.n_fft,
    )
