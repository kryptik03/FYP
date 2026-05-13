"""
process_and_isolate.py
======================
Processes Tektronix .wfm files from measured PD batches:
  1. Recursively finds all non-Noise .wfm files in data/unprocessed_measured/
  2. Converts .wfm → CSV via ConvertTekWfm.exe
  3. Denoises via Wavelet Transform (Sym4, Level 4)
  4. Isolates non-noise portions using rolling-RMS
  5. Saves isolated segments into an interim folder, grouped by batch and voltage.
     E.g., data/interim_measured/isolated_waveforms/PD2_Incision_Batch1_14kV.h5
"""

import os
import re
import sys
import glob
import shutil
import subprocess
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import h5py
import pywt
import argparse
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────

CONVERTER_EXE = SCRIPT_DIR / "ConvertTekWfm.exe"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "unprocessed_measured"
INTERIM_ROOT = PROJECT_ROOT / "data" / "interim_measured" / "isolated_waveforms"

RMS_WINDOW = 30
SIGMA_K = 0.7
MERGE_GAP_SAMPLES = 100
GUARD_MARGIN_SAMPLES = 50
MIN_REGION_SAMPLES = 10

WAVELET_TYPE = 'sym4'
WAVELET_LEVEL = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── WFM / CSV helpers ────────────────────────────────────────────

def convert_wfm_to_csv(wfm_path: Path, csv_path: Path) -> bool:
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
    max_level = pywt.dwt_max_level(data_len=len(signal), filter_len=pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)
    coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(denoised_coeffs, wavelet, mode='per')


# ─── Noise isolation ──────────────────────────────────────────────

def _rolling_rms(signal: np.ndarray, window: int) -> np.ndarray:
    sq = signal ** 2
    cs = np.cumsum(sq)
    cs = np.concatenate(([0.0], cs))
    rms = np.sqrt((cs[window:] - cs[:-window]) / window)
    pad = np.full(window - 1, rms[0])
    return np.concatenate((pad, rms))

def isolate_non_noise(signal: np.ndarray):
    signal_dc = signal - np.mean(signal)
    rms = _rolling_rms(signal_dc, RMS_WINDOW)

    baseline = np.median(rms)
    spread   = np.std(rms)
    threshold = baseline + SIGMA_K * spread

    active = rms > threshold
    if not np.any(active):
        return [], threshold, signal_dc

    diff = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    merged = []
    s, e = starts[0], ends[0]
    for ns, ne in zip(starts[1:], ends[1:]):
        if ns - e <= MERGE_GAP_SAMPLES:
            e = ne
        else:
            merged.append((s, e))
            s, e = ns, ne
    merged.append((s, e))

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

def parse_filename(wfm_path: Path):
    name = wfm_path.name
    vm = _VOLTAGE_RE.search(name)
    sm = _SAMPLE_RE.search(name)
    cm = _CHANNEL_RE.search(name)
    if not (vm and sm and cm):
        return None
    return vm.group(1).lower(), sm.group(1), cm.group(1)

# ─── HDF5 helpers ─────────────────────────────────────────────────

def save_to_hdf5(
    h5_path: Path, voltage: str, sample_id: str, channel: str,
    orig_time: np.ndarray, orig_signal: np.ndarray, denoised_signal: np.ndarray,
    iso_time: np.ndarray, iso_signal: np.ndarray,
    source_file: str, start_idx: int, end_idx: int, threshold: float
):
    with h5py.File(h5_path, "a") as f:
        grp_path = f"{voltage}/sample_{sample_id}/{channel}"
        if grp_path in f:
            del f[grp_path]
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
        grp.attrs["denoise_method"] = "wavelet"


# ─── Main Pipeline ────────────────────────────────────────────────

def process_batch(input_dir: Path, interim_dir: Path):
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    interim_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = SCRIPT_DIR / "_csv_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Recursively find all .wfm files, skipping "Noise" directories
    wfm_files = []
    for wfm in input_dir.rglob("*.wfm"):
        if "noise" in wfm.parent.name.lower() or "noise" in wfm.name.lower():
            continue
        wfm_files.append(wfm)

    if not wfm_files:
        log.error("No PD .wfm files found in %s", input_dir)
        return

    log.info("Found %d PD .wfm files in %s", len(wfm_files), input_dir)

    stats = defaultdict(lambda: {"processed": 0, "isolated": 0, "skipped": 0})
    
    # Group by batch_name (parent directory) and voltage
    batch_groups = defaultdict(lambda: defaultdict(list))
    
    for wfm in wfm_files:
        parsed = parse_filename(wfm)
        if parsed:
            voltage, _, _ = parsed
            batch_name = wfm.parent.name
            batch_groups[batch_name][voltage].append(wfm)

    for batch_name, voltage_groups in sorted(batch_groups.items()):
        for voltage, files in sorted(voltage_groups.items()):
            # e.g., PD2_Incision_Batch1_14kv.h5
            h5_path = interim_dir / f"{batch_name}_{voltage}.h5"
            log.info("── Batch: %s | Voltage: %s (%d files) → %s", batch_name, voltage, len(files), h5_path.name)

            for wfm_path in tqdm(files, desc=f"  {batch_name} {voltage}", unit="wfm", leave=False):
                parsed = parse_filename(wfm_path)
                if not parsed: continue
                _, sample_id, channel = parsed
                st = stats[f"{batch_name}_{voltage}"]

                csv_path = tmp_dir / wfm_path.with_suffix(".csv").name
                if not convert_wfm_to_csv(wfm_path, csv_path):
                    st["skipped"] += 1
                    continue

                orig_time, orig_signal = load_csv(csv_path)
                csv_path.unlink(missing_ok=True)

                if orig_time is None:
                    st["skipped"] += 1
                    continue

                st["processed"] += 1

                # Mandatory wavelet denoising
                denoised_sig = wavelet_denoise(orig_signal)

                regions, threshold, signal_dc = isolate_non_noise(denoised_sig)
                if not regions:
                    st["skipped"] += 1
                    continue

                best_region = max(regions, key=lambda r: np.mean(signal_dc[r[0]:r[1]] ** 2))
                s, e = best_region

                iso_time = orig_time[s:e]
                iso_signal = signal_dc[s:e].copy()
                
                taper_len = min(GUARD_MARGIN_SAMPLES, len(iso_signal) // 2)
                if taper_len > 0:
                    taper = np.hanning(2 * taper_len + 1)
                    iso_signal[:taper_len] *= taper[:taper_len]
                    iso_signal[-taper_len:] *= taper[-taper_len:]
                
                orig_signal_dc = orig_signal - np.mean(orig_signal)

                save_to_hdf5(
                    h5_path, voltage, sample_id, channel,
                    orig_time, orig_signal_dc, signal_dc, iso_time, iso_signal,
                    str(wfm_path), s, e, threshold
                )
                st["isolated"] += 1

    shutil.rmtree(tmp_dir, ignore_errors=True)

    log.info("\n%s  ISOLATION SUMMARY  %s", "─"*25, "─"*25)
    for group, st in sorted(stats.items()):
        log.info("  %25s  →  processed: %4d  |  isolated: %4d  |  skipped: %4d",
                 group, st["processed"], st["isolated"], st["skipped"])


def main():
    parser = argparse.ArgumentParser(description="Isolate PD waveforms from all batches in unprocessed_measured")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()

    interim_dir = INTERIM_ROOT
    
    log.info("Input    : %s", args.input_dir.resolve())
    log.info("Interim  : %s", interim_dir)
    log.info("Denoise  : wavelet (forced)")

    process_batch(args.input_dir.resolve(), interim_dir)

if __name__ == "__main__":
    main()
