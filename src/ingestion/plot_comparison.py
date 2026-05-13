"""
plot_comparison.py
==================
Loads HDF5 files produced by process_and_isolate.py and renders
side-by-side comparison plots:

  LEFT  — Full original waveform with the isolated region shaded
  RIGHT — Zoomed-in isolated portion only

Usage:
    # Plot N random samples from a specific .h5 file
    python plot_comparison.py                              # auto-finds all .h5 files
    python plot_comparison.py <path_to_h5_file>           # specific file
    python plot_comparison.py <path_to_h5_file> --n 10   # first 10 samples
    python plot_comparison.py <path_to_h5_file> --voltage 8kv --n 5
    python plot_comparison.py <path_to_h5_file> --sample 001 # plot a specific sample

Figures are saved as PNGs inside Temp_Code/Plots/<batch>/<voltage>/
and also displayed interactively (close window to advance).
"""

import argparse
import random
import sys
from pathlib import Path
from itertools import islice

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────

SCRIPT_DIR      = Path(__file__).parent.resolve()
HDF5_ROOT       = SCRIPT_DIR / "Isolated_Signals_HDF5"
PLOTS_ROOT      = SCRIPT_DIR / "Plots"

# Colour scheme per channel
CH_COLOURS = {
    "Ch1": "#4FC3F7",   # sky blue
    "Ch2": "#81C784",   # sage green
    "Ch3": "#FFB74D",   # amber
    "Ch4": "#F06292",   # rose
}
DEFAULT_COLOUR = "#CE93D8"   # lavender fallback

# Figure styling
STYLE = {
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#C9D1D9",
    "axes.titlecolor":   "#E6EDF3",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "grid.color":        "#21262D",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "text.color":        "#C9D1D9",
    "font.family":       "DejaVu Sans",
}

# ──────────────────────────────────────────────────────────────────


def _time_to_us(arr: np.ndarray) -> np.ndarray:
    """Scale time array to microseconds for readable tick labels."""
    max_abs = np.max(np.abs(arr))
    if max_abs < 1e-3:            # already in µs-ish range
        return arr * 1e6, "µs"
    elif max_abs < 1:             # milliseconds
        return arr * 1e3, "ms"
    return arr, "s"


def _channel_colour(channel: str) -> str:
    return CH_COLOURS.get(channel, DEFAULT_COLOUR)


def plot_triplet(
    orig_time:   np.ndarray,
    orig_signal: np.ndarray,
    denoised_signal: np.ndarray,
    iso_time:    np.ndarray,
    iso_signal:  np.ndarray,
    start_idx:   int,
    end_idx:     int,
    threshold:   float,
    channel:     str,
    title:       str,
    save_path:   Path | None = None,
    show:        bool = True,
):
    """
    Render one side-by-side comparison figure.

    LEFT  : Full original waveform
    MID   : Denoised waveform + shaded isolated region + threshold line
    RIGHT : Isolated portion only (zoomed in), annotated with duration & peak
    """
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(
            1, 3,
            figsize=(18, 4.5),
            gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.35},
        )
        colour = _channel_colour(channel)
        shade  = (*matplotlib.colors.to_rgb(colour), 0.18)

        ot, t_unit = _time_to_us(orig_time)

        # ── Left: full original waveform ────────────────────────
        ax0 = axes[0]
        ax0.plot(ot, orig_signal, color=colour, linewidth=0.6, alpha=0.9,
                 label="Original signal")

        ax0.set_xlabel(f"Time ({t_unit})")
        ax0.set_ylabel("Amplitude (V)")
        ax0.set_title("Original Waveform", fontsize=10, pad=6)
        ax0.xaxis.set_minor_locator(AutoMinorLocator())
        ax0.yaxis.set_minor_locator(AutoMinorLocator())
        ax0.grid(True, which="major")
        ax0.legend(fontsize=7, loc="upper right",
                   facecolor="#21262D", edgecolor="#30363D")

        # ── Middle: full denoised waveform ──────────────────────
        ax1 = axes[1]
        ax1.plot(ot, denoised_signal, color=colour, linewidth=0.6, alpha=0.9,
                 label="Denoised signal")

        # Shade the isolated region
        if start_idx < end_idx:
            ax1.axvspan(ot[start_idx], ot[min(end_idx, len(ot)-1)],
                        color=shade, label="Isolated region")

        # Noise threshold lines (±)
        ax1.axhline( threshold, color="#F85149", linewidth=0.7,
                     linestyle=":", alpha=0.8, label=f"+threshold ({threshold:.3g} V)")
        ax1.axhline(-threshold, color="#F85149", linewidth=0.7,
                     linestyle=":", alpha=0.8)

        ax1.set_xlabel(f"Time ({t_unit})")
        ax1.set_title("Denoised Waveform", fontsize=10, pad=6)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.grid(True, which="major")
        ax1.legend(fontsize=7, loc="upper right",
                   facecolor="#21262D", edgecolor="#30363D")

        # ── Right: isolated zoom ────────────────────────────────
        ax2 = axes[2]
        it, _ = _time_to_us(iso_time)

        # Shift isolated time to start at zero for clearer view
        it_shifted = it - it[0]

        ax2.fill_between(it_shifted, iso_signal, alpha=0.25, color=colour)
        ax2.plot(it_shifted, iso_signal, color=colour, linewidth=0.9)

        peak_v   = np.max(np.abs(iso_signal))
        duration = it[-1] - it[0]
        n_pts    = len(iso_signal)

        ax2.set_xlabel(f"Relative time ({t_unit})")
        ax2.set_ylabel("Amplitude (V)")
        ax2.set_title("Isolated region (zoomed)", fontsize=10, pad=6)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.grid(True, which="major")

        # Annotation box
        info = (
            f"Duration : {duration:.3g} {t_unit}\n"
            f"Peak     : {peak_v:.4g} V\n"
            f"Points   : {n_pts}"
        )
        ax2.text(
            0.97, 0.97, info,
            transform=ax2.transAxes,
            fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="#21262D",
                      ec="#30363D", alpha=0.85),
        )

        # ── Shared figure title ─────────────────────────────────
        fig.suptitle(title, fontsize=11, fontweight="bold",
                     color="#E6EDF3", y=1.01)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  -> Saved: {save_path}")

        if show:
            plt.show()

        plt.close(fig)


def iter_h5_entries(h5_file: Path, voltage_filter: str | None = None, sample_filter: str | None = None):
    """
    Yield dicts of waveform data for every (voltage, sample, channel)
    group found in the HDF5 file.
    """
    with h5py.File(h5_file, "r") as f:
        for voltage in f:
            if voltage_filter and voltage.lower() != voltage_filter.lower():
                continue
            for sample in f[voltage]:
                if sample_filter and sample != f"sample_{sample_filter}":
                    continue
                for channel in f[voltage][sample]:
                    grp = f[voltage][sample][channel]
                    try:
                        # Fallback for old files without denoised_signal
                        denoised = grp["denoised_signal"][:] if "denoised_signal" in grp else grp["orig_signal"][:]
                        yield {
                            "voltage":     voltage,
                            "sample":      sample,
                            "channel":     channel,
                            "orig_time":   grp["orig_time"][:],
                            "orig_signal": grp["orig_signal"][:],
                            "denoised_signal": denoised,
                            "iso_time":    grp["iso_time"][:],
                            "iso_signal":  grp["iso_signal"][:],
                            "start_idx":   int(grp.attrs.get("start_idx", 0)),
                            "end_idx":     int(grp.attrs.get("end_idx", 0)),
                            "threshold":   float(grp.attrs.get("threshold", 0)),
                            "source_file": str(grp.attrs.get("source_file", "")),
                        }
                    except Exception as exc:
                        print(f"  [warn] Skipping {voltage}/{sample}/{channel}: {exc}")


def find_h5_files(root: Path) -> list[Path]:
    """Recursively find all .h5 files under root."""
    return sorted(root.rglob("*.h5"))


def plot_h5_file(
    h5_path:        Path,
    n:              int,
    voltage_filter: str | None,
    sample_filter:  str | None,
    save_plots:     bool,
    show_plots:     bool,
    random_sample:  bool,
):
    """Load an HDF5 file and plot n entries."""
    print(f"\n-- Reading: {h5_path.name}")

    # Collect all entries into a list (needed for random sampling)
    entries = list(iter_h5_entries(h5_path, voltage_filter, sample_filter))

    if not entries:
        print("  No entries found (check voltage/sample filter?).")
        return

    print(f"  Found {len(entries)} waveform entries.")

    if random_sample and not sample_filter:
        selected = random.sample(entries, min(n, len(entries)))
    elif sample_filter:
        selected = entries # Take all channels for this sample
    else:
        selected = entries[:n]

    # Build a consistent plot output subfolder
    batch_name  = h5_path.parent.name          # e.g. PD3_Delam_Batch2
    voltage_tag = h5_path.stem.split("_")[-1]  # e.g. 7kv
    plot_dir    = PLOTS_ROOT / batch_name / voltage_tag

    for idx, entry in enumerate(selected, start=1):
        ch      = entry["channel"]
        sample  = entry["sample"]
        voltage = entry["voltage"]
        title   = (
            f"{batch_name}  |  {voltage}  |  {sample}  |  {ch}"
        )

        save_path = None
        if save_plots:
            fname = f"{sample}_{ch}.png"
            save_path = plot_dir / fname

        print(f"  [{idx}/{len(selected)}] Plotting {voltage}/{sample}/{ch}...")
        plot_triplet(
            orig_time   = entry["orig_time"],
            orig_signal = entry["orig_signal"],
            denoised_signal = entry["denoised_signal"],
            iso_time    = entry["iso_time"],
            iso_signal  = entry["iso_signal"],
            start_idx   = entry["start_idx"],
            end_idx     = entry["end_idx"],
            threshold   = entry["threshold"],
            channel     = ch,
            title       = title,
            save_path   = save_path,
            show        = show_plots,
        )


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot original vs isolated waveforms from HDF5 files."
    )
    p.add_argument(
        "h5_file", nargs="?", type=Path, default=None,
        help="Path to a specific .h5 file. If omitted, all files in "
             f"{HDF5_ROOT} are plotted."
    )
    p.add_argument(
        "--n", type=int, default=5,
        help="Number of waveforms to plot per HDF5 file (default: 5)."
    )
    p.add_argument(
        "--voltage", type=str, default=None,
        help="Only plot this voltage level (e.g. '7kv', '10kv')."
    )
    p.add_argument(
        "--sample", type=str, default=None,
        help="Only plot this specific sample ID (e.g. '001'). If specified, ignores --n."
    )
    p.add_argument(
        "--random", action="store_true",
        help="Randomly sample entries instead of taking the first N."
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Do not save PNG figures."
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="Do not display interactive plot window (useful for batch runs)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    save_plots  = not args.no_save
    show_plots  = not args.no_show

    if args.h5_file:
        h5_files = [args.h5_file.resolve()]
    else:
        h5_files = find_h5_files(HDF5_ROOT)
        if not h5_files:
            print(f"No .h5 files found under {HDF5_ROOT}.")
            print("Run process_and_isolate.py first to generate HDF5 data.")
            sys.exit(1)
        print(f"Auto-discovered {len(h5_files)} HDF5 file(s).")

    for h5_path in h5_files:
        plot_h5_file(
            h5_path        = h5_path,
            n              = args.n,
            voltage_filter = args.voltage,
            sample_filter  = args.sample,
            save_plots     = save_plots,
            show_plots     = show_plots,
            random_sample  = args.random,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
