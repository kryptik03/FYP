"""
extract_bispectra_v2.py
=======================
V2 of the bispectrum feature extractor — produces 2-channel complex-aware
features (Magnitude + Phase) for Experiment 09.

KEY CHANGE vs extract_bispectra.py (V1)
-----------------------------------------
V1 saved: pulses_bispectra : float32 (N_pulses, 129, 129)  — magnitude only
V2 saves: pulses_bispectra : float32 (N_pulses, 2, 129, 129)
          Channel 0 = np.abs(B)   — magnitude (energy envelope)
          Channel 1 = np.angle(B) — phase in radians, range [-π, π]

WHY PRESERVE PHASE?
-------------------
np.abs() reduces the bispectrum to a pure energy envelope, discarding all
phase-coupling information. While magnitude is easily distorted by sensor
distance (attenuation), the RELATIVE PHASE between frequency components is
a stable "physics fingerprint" of the discharge mechanism. PD types whose
spectral envelopes overlap often have distinct phase patterns.

Preserving phase provides the additional discriminative axis needed for
Exp09's Distance-Invariant SupCon and 4-Domain DANN objectives.

USAGE
-----
    # Synthesised data
    python src/features/extract_bispectra_v2.py \\
        --input_dir  data/raw/synthesised/<folder> \\
        --parent_node_id <node_id>

    # CWRU data
    python src/features/extract_bispectra_v2.py \\
        --input_dir  data/raw/cwru/<folder> \\
        --parent_node_id <node_id> \\
        --n_fft 2048

Output goes to: data/features/bispectra_v2/<timestamp>-<origin>-<root_id>-<node_id>/

The output directory name is registered in src/utils/lineage.db.

OUTPUT H5 SCHEMA (V2)
---------------------
    pulses_bispectra : float32  (N_pulses, 2, F, F)
                       [:, 0, :, :] = magnitude  (np.abs)
                       [:, 1, :, :] = phase       (np.angle, radians)
                       F = nperseg // 2 + 1 = 129
    labels           : copied verbatim from input shard
    root attrs       : copied verbatim from input shard
    Added attrs:
        feature_type         = "bispectra_complex_per_pulse"
        bispectrum_channels  = ["magnitude", "phase"]
        bispectrum_version   = 2
        (all V1 attrs also present: padded_len, nperseg, noverlap, freq_bins, L)
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
# Label row indices (shared schema across all experiments)
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
# Core: 1D Pad / Crop (unchanged from V1)
# ---------------------------------------------------------------------------

def pad_or_crop_1d(signal: np.ndarray, n_fft: int) -> np.ndarray:
    """
    Force the raw 1D pulse waveform to exactly `n_fft` points.
    Shorter pulses → zero-padded at the end (causal).
    Longer pulses  → first n_fft samples kept (truncate).
    """
    signal = signal.astype(np.float32)
    L = len(signal)
    if L == n_fft:
        return signal
    if L < n_fft:
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[:L] = signal
        return padded
    return signal[:n_fft]


# ---------------------------------------------------------------------------
# Core: 2D Complex Bispectrum (Welch segment-averaging)
#       *** KEY V2 CHANGE: returns (2, F, F) instead of (F, F) ***
# ---------------------------------------------------------------------------

_NPERSEG  = 256
_NOVERLAP = 128


def compute_bispectrum_welch_complex(
    signal: np.ndarray,
    nperseg: int = 256,
    noverlap: int = 128,
    L: int = 1,
) -> np.ndarray:
    """
    Compute the 2-channel bispectrum feature map for a single isolated pulse.

    Returns a (2, F, F) float32 array where F = nperseg // 2 + 1 = 129:
        [0, :, :] = Magnitude  = np.abs(B(f1, f2))
        [1, :, :] = Phase      = np.angle(B(f1, f2))  in radians [-π, π]

    The intermediate complex bispectrum B is computed identically to V1
    (Welch segment-averaging + frequency-domain smoothing, Equations 7–8
    from Kim & Powers 1979). Only the final step differs: instead of
    discarding phase with np.abs, we save both components.

    Args:
        signal  : 1D float32 array of fixed length (n_fft points).
        nperseg : Welch segment length (default 256 → 129 frequency bins).
        noverlap: Overlap between segments (default 128).
        L       : Smoothing window radius for Eq 7 (default 1 → 3×3 filter).

    Returns:
        np.ndarray of shape (2, 129, 129), dtype float32.
    """
    step   = nperseg - noverlap
    starts = np.arange(0, len(signal) - nperseg + 1, step)

    n_bins = nperseg // 2 + 1            # 129
    bispectrum_avg = np.zeros((n_bins, n_bins), dtype=np.complex64)

    f1_idx  = np.arange(n_bins, dtype=np.int32)
    f2_idx  = np.arange(n_bins, dtype=np.int32)
    f12_idx = np.clip(f1_idx[:, None] + f2_idx[None, :], 0, n_bins - 1)

    for start in starts:
        segment = signal[start : start + nperseg]
        segment = segment - np.mean(segment)        # zero-mean (Eq assumption)
        segment = segment * np.hanning(nperseg)
        X       = np.fft.rfft(segment, n=nperseg)

        raw_b = X[f1_idx[:, None]] * X[f2_idx[None, :]] * np.conj(X[f12_idx])
        bispectrum_avg += raw_b

    # Equation 8: average across segments
    bispectrum_avg /= max(len(starts), 1)

    # Equation 7: frequency-domain smoothing
    if L > 0:
        window_size = 2 * L + 1
        real_sm = scipy.ndimage.uniform_filter(bispectrum_avg.real, size=window_size)
        imag_sm = scipy.ndimage.uniform_filter(bispectrum_avg.imag, size=window_size)
        bispectrum_avg = real_sm + 1j * imag_sm

    # *** V2 KEY CHANGE: save both magnitude and phase ***
    magnitude = np.abs(bispectrum_avg).astype(np.float32)     # (129, 129)
    phase     = np.angle(bispectrum_avg).astype(np.float32)   # (129, 129), [-π, π]

    return np.stack([magnitude, phase], axis=0)  # (2, 129, 129)


# ---------------------------------------------------------------------------
# Shard Processing
# ---------------------------------------------------------------------------

def process_shard(input_path: str, output_path: str, n_fft: int):
    """
    Read one raw 1D shard, compute a 2-channel complex bispectrum for every
    individual PD pulse, and write the result to a new V2 output shard.

    H5 Input Schema (raw shards — unchanged):
        scenes : float32  (N_scenes, N_channels, T_samples)
        labels : float32  (7, N_pulses)
        attrs  : sampling_frequency_Hz, time_resolution_s, ...

    H5 Output Schema (V2 bispectrum feature shards):
        pulses_bispectra : float32 (N_pulses, 2, F, F)
                           [:, 0, :, :] = magnitude
                           [:, 1, :, :] = phase (radians)
                           F = _NPERSEG // 2 + 1 = 129
        labels           : copied verbatim
        attrs            : all original attrs + V2 bispectrum metadata
    """
    n_bins = _NPERSEG // 2 + 1   # 129

    with h5py.File(input_path, "r") as h5_in:
        if "scenes" not in h5_in or "labels" not in h5_in:
            print(f"    [Skip] Missing 'scenes' or 'labels' in {input_path}")
            return

        scenes_1d = h5_in["scenes"][:]   # (N_scenes, N_channels, T_samples)
        labels    = h5_in["labels"][:]   # (7, N_pulses)

        N_scenes, N_channels, T_samples = scenes_1d.shape
        N_pulses = labels.shape[1]

        print(f"    Input : {N_scenes} scenes × {N_channels} ch × {T_samples} pts, "
              f"{N_pulses} pulses")

        # *** V2: allocate (N_pulses, 2, F, F) ***
        pulses_bispectra = np.zeros(
            (N_pulses, 2, n_bins, n_bins), dtype=np.float32
        )

        for k in range(N_pulses):
            scene_idx = int(labels[ROW_SCENE_ID,   k])
            ch_idx    = int(labels[ROW_CHANNEL_ID, k])
            start_idx = int(labels[ROW_START_IDX,  k])
            end_idx   = int(labels[ROW_END_IDX,    k])

            raw_pulse     = scenes_1d[scene_idx, ch_idx, start_idx : end_idx + 1]
            uniform_pulse = pad_or_crop_1d(raw_pulse, n_fft)

            pulses_bispectra[k] = compute_bispectrum_welch_complex(
                uniform_pulse, nperseg=_NPERSEG, noverlap=_NOVERLAP
            )

        labels_data  = labels
        labels_attrs = dict(h5_in["labels"].attrs)
        root_attrs   = dict(h5_in.attrs)

    with h5py.File(output_path, "w") as h5_out:
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

        # V2-specific metadata
        h5_out.attrs["bispectrum_padded_len"]  = n_fft
        h5_out.attrs["bispectrum_nperseg"]     = _NPERSEG
        h5_out.attrs["bispectrum_noverlap"]    = _NOVERLAP
        h5_out.attrs["bispectrum_freq_bins"]   = n_bins
        h5_out.attrs["bispectrum_L"]           = 1
        h5_out.attrs["bispectrum_version"]     = 2
        h5_out.attrs["feature_type"]           = "bispectra_complex_per_pulse"
        # Store channel names as a JSON string (HDF5 attrs don't support lists natively)
        h5_out.attrs["bispectrum_channels"]    = '["magnitude", "phase"]'

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
    Process every shard in `input_dir` and write V2 per-pulse 2-channel
    bispectra to a new timestamped folder under `output_root/bispectra_v2/`.
    Registers the run in the SQLite lineage database.
    """
    node_id   = generate_short_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Query lineage DB for parent metadata
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

    # Output folder: data/features/bispectra_v2/<ts>-<origin>-<root_id>-<node_id>/
    folder_name       = f"{timestamp}-{origin}-{root_id}-{node_id}"
    target_output_dir = os.path.join(output_root, "bispectra_v2", folder_name)
    os.makedirs(target_output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    if not h5_files:
        print(f"[Error] No .h5 files found in: {input_dir}")
        return

    print("=" * 60)
    print("  Bispectrum V2 Feature Extraction  (COMPLEX — PER-PULSE)")
    print(f"  Input    : {input_dir}")
    print(f"  Output   : {target_output_dir}")
    print(f"  n_fft    : {n_fft}  (pad/crop target per pulse)")
    print(f"  nperseg  : {_NPERSEG}  ->  F bins = {_NPERSEG // 2 + 1}")
    print(f"  Channels : [0=magnitude, 1=phase]")
    print(f"  Shards   : {len(h5_files)}")
    print("=" * 60)

    for h5_file in h5_files:
        fname = os.path.basename(h5_file)
        out_f = os.path.join(target_output_dir, fname)
        print(f"\n[Shard] {fname}")
        process_shard(h5_file, out_f, n_fft)

    history_log = (
        f"Per-pulse 2-channel (magnitude+phase) bispectrum extraction (V2). "
        f"n_fft={n_fft}, nperseg={_NPERSEG}, noverlap={_NOVERLAP}, "
        f"freq_bins={_NPERSEG // 2 + 1}. Output shape=(N,2,{_NPERSEG // 2 + 1},{_NPERSEG // 2 + 1}). "
        f"Shards={len(h5_files)}."
    )

    if register_process is not None:
        register_process(
            parent_id        = parent_node_id,
            stage            = "feature_extraction",
            method           = "bispectrum_complex_per_pulse_v2",
            folder_path      = target_output_dir,
            appended_history = history_log,
            force_node_id    = node_id,
            force_timestamp  = timestamp,
        )
        print(f"\n[Lineage] Registered node {node_id} (child of {parent_node_id}).")

    print(f"\n[Done] V2 complex bispectra saved to: {target_output_dir}")
    return target_output_dir


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-pulse 2-channel (magnitude + phase) bispectrum features "
            "from raw 1D H5 shards. V2 for Experiment 09."
        )
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing raw .h5 shards ('scenes' + 'labels' datasets).",
    )
    parser.add_argument(
        "--output_root", type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "features")
        ),
        help="Root dir for feature outputs. Subdirectory 'bispectra_v2/' will be created.",
    )
    parser.add_argument(
        "--parent_node_id", type=str, required=True,
        help="Lineage node ID of the raw dataset (for DAG registration).",
    )
    parser.add_argument(
        "--n_fft", type=int, default=4096,
        help="Target length for pad/crop of each isolated pulse (default: 4096).",
    )

    args = parser.parse_args()
    extract_features(
        input_dir      = os.path.abspath(args.input_dir),
        output_root    = os.path.abspath(args.output_root),
        parent_node_id = args.parent_node_id,
        n_fft          = args.n_fft,
    )
