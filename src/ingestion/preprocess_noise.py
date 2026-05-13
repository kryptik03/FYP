"""
preprocess_noise.py
===================
Converts all .wfm files from measured Noise folders into a single HDF5 database
to allow fast, O(1) random sampling during shard generation.

Output: data/noise/real_noise_db.h5
"""

import os
import sys
import glob
import subprocess
import tempfile
import argparse
import logging
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()
CONVERTER_EXE = SCRIPT_DIR / "ConvertTekWfm.exe"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def convert_wfm_to_csv(wfm_path: Path, csv_path: Path) -> bool:
    if not CONVERTER_EXE.is_file():
        log.error("ConvertTekWfm.exe not found at %s", CONVERTER_EXE)
        return False
    try:
        result = subprocess.run(
            [str(CONVERTER_EXE), str(wfm_path), "/CSV", str(csv_path)],
            capture_output=True, timeout=60
        )
        return result.returncode == 0
    except Exception as exc:
        log.warning("Converter raised %s for %s", exc, wfm_path.name)
        return False


def load_csv_amplitude(csv_path: Path):
    """Parse the Tektronix CSV export, but only keep amplitude (col 4)."""
    try:
        data = np.genfromtxt(csv_path, delimiter=",", usecols=(4,))
        return data.astype(np.float32)
    except Exception as exc:
        log.warning("CSV parse error (%s): %s", csv_path.name, exc)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=PROJECT_ROOT / "data" / "unprocessed_measured",
                        help="Root directory containing Noise folders.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "noise" / "real_noise_db.h5",
                        help="Path to output HDF5 database.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    # Find all Noise .wfm files
    noise_wfm_files = []
    for noise_dir in ["Noise_10M", "Noise_RL12.5M"]:
        target_dir = input_dir / noise_dir
        if target_dir.is_dir():
            noise_wfm_files.extend(list(target_dir.glob("*.wfm")))

    if not noise_wfm_files:
        log.error("No Noise .wfm files found in %s", input_dir)
        sys.exit(1)

    MAX_NOISE_FILES = 20
    if len(noise_wfm_files) > MAX_NOISE_FILES:
        import random
        random.seed(42)
        noise_wfm_files = random.sample(noise_wfm_files, MAX_NOISE_FILES)

    log.info("Found %d noise .wfm files to process.", len(noise_wfm_files))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        log.info("Removing existing noise DB: %s", output_path)
        output_path.unlink()

    tmp_dir = SCRIPT_DIR / "_csv_tmp"
    tmp_dir.mkdir(exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        trace_grp = h5f.create_group("traces")
        
        trace_idx = 0
        for wfm_path in tqdm(noise_wfm_files, desc="Converting noise files"):
            csv_path = tmp_dir / wfm_path.with_suffix(".csv").name
            ok = convert_wfm_to_csv(wfm_path, csv_path)
            if not ok:
                continue
            
            amp_data = load_csv_amplitude(csv_path)
            csv_path.unlink(missing_ok=True)

            if amp_data is None or len(amp_data) < 50000:
                log.warning("Skipping %s (too short or parse failed).", wfm_path.name)
                continue

            # Save to HDF5
            ds_name = f"trace_{trace_idx:04d}"
            ds = trace_grp.create_dataset(ds_name, data=amp_data, compression="gzip", compression_opts=4)
            ds.attrs["source_file"] = wfm_path.name
            ds.attrs["length"] = len(amp_data)
            
            trace_idx += 1

    # Cleanup tmp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    log.info("Successfully converted %d noise traces.", trace_idx)
    log.info("Saved to: %s", output_path)


if __name__ == "__main__":
    main()
