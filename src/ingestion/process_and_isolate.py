"""
process_and_isolate.py
======================
Processes Tektronix .wfm files from a PD3 Delamination batch folder:
  1. Converts each .wfm → temporary CSV via ConvertTekWfm.exe
  2. Intelligently isolates the non-noise portion of each waveform
     using an adaptive rolling-RMS noise floor + sigma thresholding
  3. Saves both the full original and isolated signals into HDF5 files
     organized by voltage level (one .h5 file per voltage)

Output: Temp_Code/Isolated_Signals_HDF5/<batch>/PD3_Delam_<voltage>.h5

Usage:
    python process_and_isolate.py                    # uses INPUT_DIR below
    python process_and_isolate.py <path_to_batch>    # custom batch folder
"""

import os
import re
import sys
import glob
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import pywt
import argparse
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these as needed)
# ──────────────────────────────────────────────────────────────────

# Root of this script — always Temp_Code/
SCRIPT_DIR = Path(__file__).parent.resolve()

# Path to the ConvertTekWfm executable (lives in Temp_Code/)
CONVERTER_EXE = SCRIPT_DIR / "ConvertTekWfm.exe"

# Default input: change to Batch3 path when that data is collected
DEFAULT_INPUT_DIR = (
    SCRIPT_DIR.parent / "PD3_Delamination" / "PD3_Delam_Batch2"
)

# All HDF5 output lands inside Temp_Code/Isolated_Signals_HDF5/
HDF5_OUTPUT_ROOT = SCRIPT_DIR / "Isolated_Signals_HDF5"

# ── Noise isolation tuning ──────────────────────────────────────
# Window length (samples) for the rolling-RMS envelope
RMS_WINDOW = 30

# Threshold multiplier: threshold = median_rms + SIGMA_K × std_rms
# Higher → stricter (fewer false positives); lower → more sensitive
SIGMA_K = 0.7

# Merge isolated burst regions if the gap between them is ≤ this
MERGE_GAP_SAMPLES = 100

# Guard margin to pad each detected region on both sides
GUARD_MARGIN_SAMPLES = 50

# Minimum length of an isolated region to be kept (samples)
MIN_REGION_SAMPLES = 10

# ── Wavelet Denoising tuning ──────────────────────────────────────
# Default wavelet to use for PD signals
WAVELET_TYPE = 'sym4'
# Decomposition level
WAVELET_LEVEL = 4

# ──────────────────────────────────────────────────────────────────


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── WFM / CSV helpers ────────────────────────────────────────────

def convert_wfm_to_csv(wfm_path: Path, csv_path: Path) -> bool:
    """
    Invoke ConvertTekWfm.exe to convert a .wfm file to CSV.
    The converter is always read from SCRIPT_DIR (Temp_Code/).
    The output CSV is written to csv_path (also inside Temp_Code/).

    Returns True on success, False on failure.
    """
    if not CONVERTER_EXE.is_file():
        log.error("ConvertTekWfm.exe not found at %s", CONVERTER_EXE)
        return False
    try:
        result = subprocess.run(
            [str(CONVERTER_EXE), str(wfm_path), "/CSV", str(csv_path)],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except Exception as exc:
        log.warning("Converter raised %s for %s", exc, wfm_path.name)
        return False


def load_csv(csv_path: Path):
    """
    Parse the Tektronix CSV export.
    Columns 3 and 4 (0-indexed) carry time and amplitude.

    Returns (time_array, amplitude_array) or (None, None) on failure.
    """
    try:
        data = np.genfromtxt(csv_path, delimiter=",", usecols=(3, 4))
        if data.ndim < 2 or data.shape[0] < 2:
            return None, None
        return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)
    except Exception as exc:
        log.warning("CSV parse error (%s): %s", csv_path.name, exc)
        return None, None


# ─── Wavelet Denoising ────────────────────────────────────────────

def wavelet_denoise(signal: np.ndarray, wavelet: str = WAVELET_TYPE, level: int = WAVELET_LEVEL) -> np.ndarray:
    """
    Denoise a signal using Discrete Wavelet Transform and soft thresholding.
    Uses universal thresholding (MAD) adapted for PD signals.
    """
    # Calculate maximum decomposition level to avoid errors with short signals
    max_level = pywt.dwt_max_level(data_len=len(signal), filter_len=pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)
    
    # Decompose to get coefficients
    coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)
    
    # Calculate universal threshold based on MAD of the detail coefficients (highest frequency)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Apply soft thresholding to detail coefficients
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    
    # Reconstruct signal
    return pywt.waverec(denoised_coeffs, wavelet, mode='per')


# ─── Noise isolation ──────────────────────────────────────────────

def _rolling_rms(signal: np.ndarray, window: int) -> np.ndarray:
    """Compute a causal rolling RMS using cumulative-sum trick (fast)."""
    sq = signal ** 2
    cs = np.cumsum(sq)
    cs = np.concatenate(([0.0], cs))
    rms = np.sqrt((cs[window:] - cs[:-window]) / window)
    # Pad the start so length matches input
    pad = np.full(window - 1, rms[0])
    return np.concatenate((pad, rms))


def isolate_non_noise(signal: np.ndarray):
    """
    Detect the non-noise portion(s) of a waveform using an adaptive
    rolling-RMS noise floor with sigma thresholding.

    Algorithm
    ---------
    1. DC-remove the signal (subtract mean).
    2. Compute rolling RMS with window = RMS_WINDOW.
    3. Estimate noise baseline = median(rms) and spread = std(rms).
    4. Threshold = baseline + SIGMA_K × spread.
    5. Find contiguous spans where rms > threshold.
    6. Merge spans whose gap ≤ MERGE_GAP_SAMPLES.
    7. Pad each span by GUARD_MARGIN_SAMPLES on both sides.
    8. Discard spans shorter than MIN_REGION_SAMPLES.

    Returns
    -------
    regions : list of (start_idx, end_idx) — may be empty if no PD detected
    threshold : float — the computed threshold value (for metadata)
    signal_dc : np.ndarray — DC-removed signal
    """
    signal_dc = signal - np.mean(signal)
    rms = _rolling_rms(signal_dc, RMS_WINDOW)

    baseline = np.median(rms)
    spread   = np.std(rms)
    threshold = baseline + SIGMA_K * spread

    # Boolean mask: where is activity above noise?
    active = rms > threshold

    if not np.any(active):
        return [], threshold, signal_dc

    # Convert mask to (start, end) index pairs for each contiguous run
    diff = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]   # exclusive

    # Merge runs that are close together
    merged = []
    s, e = starts[0], ends[0]
    for ns, ne in zip(starts[1:], ends[1:]):
        if ns - e <= MERGE_GAP_SAMPLES:
            e = ne          # extend current region
        else:
            merged.append((s, e))
            s, e = ns, ne
    merged.append((s, e))

    # Pad with guard margin and clip to signal bounds
    n = len(signal_dc)
    regions = []
    for s, e in merged:
        s = max(0, s - GUARD_MARGIN_SAMPLES)
        e = min(n, e + GUARD_MARGIN_SAMPLES)
        if (e - s) >= MIN_REGION_SAMPLES:
            regions.append((int(s), int(e)))

    return regions, threshold, signal_dc


# ─── Filename parsing ─────────────────────────────────────────────

_VOLTAGE_RE  = re.compile(r"_(\d+kV)_", re.IGNORECASE)
_SAMPLE_RE   = re.compile(r"_(\d{3})_")
_CHANNEL_RE  = re.compile(r"_(Ch\d+)\.wfm", re.IGNORECASE)
_PD_NUMBER_AND_TYPE   = re.compile(r"(PD\d+_.+?)_")


def parse_filename(wfm_path: Path):
    """
    Extract (voltage_str, sample_str, channel_str) from a filename like
    'PD3_Delam_7kV_Batch2_000_Ch1.wfm'.

    Returns None if any component cannot be parsed.
    """
    name = wfm_path.name
    vm = _VOLTAGE_RE.search(name)
    sm = _SAMPLE_RE.search(name)
    cm = _CHANNEL_RE.search(name)
    pdm = _PD_NUMBER_AND_TYPE.search(name)
    if not (vm and sm and cm and pdm):
        return None
    return vm.group(1).lower(), sm.group(1), cm.group(1), pdm.group(1)


# ─── HDF5 helpers ─────────────────────────────────────────────────

def save_to_hdf5(
    h5_path: Path,
    voltage: str,
    sample_id: str,
    channel: str,
    orig_time:   np.ndarray,
    orig_signal: np.ndarray,
    denoised_signal: np.ndarray,
    iso_time:    np.ndarray,
    iso_signal:  np.ndarray,
    source_file: str,
    start_idx:   int,
    end_idx:     int,
    threshold:   float,
    denoise_method: str,
):
    """Append one isolated segment to the appropriate HDF5 file."""
    with h5py.File(h5_path, "a") as f:
        grp_path = f"{voltage}/sample_{sample_id}/{channel}"
        if grp_path in f:
            del f[grp_path]         # overwrite if re-processing
        grp = f.require_group(grp_path)

        grp.create_dataset("orig_time",   data=orig_time,   compression="gzip")
        grp.create_dataset("orig_signal", data=orig_signal, compression="gzip")
        grp.create_dataset("denoised_signal", data=denoised_signal, compression="gzip")
        grp.create_dataset("iso_time",    data=iso_time,    compression="gzip")
        grp.create_dataset("iso_signal",  data=iso_signal,  compression="gzip")

        grp.attrs["source_file"] = source_file
        grp.attrs["start_idx"]   = start_idx
        grp.attrs["end_idx"]     = end_idx
        grp.attrs["threshold"]   = threshold
        grp.attrs["rms_window"]  = RMS_WINDOW
        grp.attrs["sigma_k"]     = SIGMA_K
        grp.attrs["denoise_method"] = denoise_method


# ─── Main ─────────────────────────────────────────────────────────

def process_batch(input_dir: Path, output_dir: Path, denoise_method: str):
    """Process all .wfm files in input_dir and write HDF5 to output_dir."""
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a private temp folder inside Temp_Code for intermediate CSVs
    tmp_dir = SCRIPT_DIR / "_csv_tmp"
    tmp_dir.mkdir(exist_ok=True)

    wfm_files = sorted(input_dir.glob("*.wfm"))
    if not wfm_files:
        log.error("No .wfm files found in %s", input_dir)
        return

    batch_name = input_dir.name           # e.g. "PD3_Delam_Batch2"
    batch_output = output_dir / batch_name
    batch_output.mkdir(parents=True, exist_ok=True)

    log.info("Found %d .wfm files in %s", len(wfm_files), input_dir)
    log.info("Output → %s", batch_output)

    # Stats
    stats = defaultdict(lambda: {"processed": 0, "isolated": 0, "skipped": 0})

    # Group files by voltage for progress display
    voltage_groups = defaultdict(lambda: defaultdict(list))
    unparseable = []
    for wfm in wfm_files:
        parsed = parse_filename(wfm)
        if parsed is None:
            unparseable.append(wfm)
            continue
        voltage, _, _, pd_type = parsed
        voltage_groups[voltage][pd_type].append(wfm)

    if unparseable:
        log.warning("%d files could not be parsed and will be skipped.",
                    len(unparseable))

    for voltage, pd_types in sorted(voltage_groups.items()):
        for pd_type, files in sorted(pd_types.items()):
            h5_path = batch_output / f"{pd_type}_{voltage}.h5"
            log.info("── Voltage level: %s  (%d files) → %s",
                     voltage, len(files), h5_path.name)

            for wfm_path in tqdm(files, desc=f"  {voltage}", unit="wfm",
                              ncols=80, leave=False):
                parsed = parse_filename(wfm_path)
                if parsed is None:
                    continue
                _, sample_id, channel, pd_type = parsed
                st = stats[voltage]

                # 1. Convert .wfm → temporary CSV (output in Temp_Code/)
                csv_path = tmp_dir / wfm_path.with_suffix(".csv").name
                ok = convert_wfm_to_csv(wfm_path, csv_path)
                if not ok:
                    log.debug("Conversion failed: %s", wfm_path.name)
                    st["skipped"] += 1
                    continue

                # 2. Load CSV
                orig_time, orig_signal = load_csv(csv_path)
                csv_path.unlink(missing_ok=True)   # clean up immediately

                if orig_time is None:
                    st["skipped"] += 1
                    continue

                st["processed"] += 1

                # 3. Apply selected denoising method
                if denoise_method == 'wavelet':
                    denoised_sig = wavelet_denoise(orig_signal)
                else:
                    denoised_sig = orig_signal

                # 4. Isolate non-noise region(s) on the DENOISED signal
                regions, threshold, signal_dc = isolate_non_noise(denoised_sig)

                if not regions:
                    log.debug("No signal detected in %s", wfm_path.name)
                    st["skipped"] += 1
                    continue

                # 5. If multiple regions, take the one with the highest RMS energy
                #    (most likely the true PD burst rather than a noise glitch)
                best_region = max(
                    regions,
                    key=lambda r: np.mean(signal_dc[r[0]:r[1]] ** 2)
                )
                s, e = best_region

                iso_time   = orig_time[s:e]
                iso_signal = signal_dc[s:e].copy()      # isolated portion of DC-removed denoised signal
                
                # Apply a smooth taper to return to zero at the padded ends
                taper_len = min(GUARD_MARGIN_SAMPLES, len(iso_signal) // 2)
                if taper_len > 0:
                    taper = np.hanning(2 * taper_len + 1)
                    fade_in = taper[:taper_len]
                    fade_out = taper[-taper_len:]
                    iso_signal[:taper_len] *= fade_in
                    iso_signal[-taper_len:] *= fade_out
                
                # Also DC-remove the full original for fair comparison
                orig_signal_dc = orig_signal - np.mean(orig_signal)

                # 6. Save to HDF5
                save_to_hdf5(
                    h5_path    = h5_path,
                    voltage    = voltage,
                    sample_id  = sample_id,
                    channel    = channel,
                    orig_time  = orig_time,
                    orig_signal= orig_signal_dc,
                    denoised_signal= signal_dc,
                    iso_time   = iso_time,
                    iso_signal = iso_signal,
                    source_file= str(wfm_path),
                    start_idx  = s,
                    end_idx    = e,
                    threshold  = threshold,
                    denoise_method = denoise_method,
                )
                st["isolated"] += 1

    # Clean up tmp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Summary
    log.info("\n%s  SUMMARY  %s", "─"*30, "─"*30)
    for voltage, st in sorted(stats.items()):
        log.info(
            "  %6s  →  processed: %4d  |  isolated: %4d  |  skipped: %4d",
            voltage, st["processed"], st["isolated"], st["skipped"]
        )
    log.info("HDF5 files written to: %s", batch_output)


def main():
    parser = argparse.ArgumentParser(description="Process Tektronix .wfm files and isolate PD signals.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Path to the batch folder containing .wfm files.")
    parser.add_argument("--denoise_method", type=str, choices=["rolling_rms", "wavelet"], default="rolling_rms",
                        help="Method to use for denoising before isolation (default: rolling_rms).")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = HDF5_OUTPUT_ROOT

    log.info("Input  : %s", input_dir)
    log.info("Output : %s", output_dir)
    log.info("Denoise: %s", args.denoise_method)

    process_batch(input_dir, output_dir, args.denoise_method)

if __name__ == "__main__":
    main()
