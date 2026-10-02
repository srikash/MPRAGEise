#!/usr/bin/env python3
"""
MPRAGEise.py

This script processes MP2RAGE images by either removing or reintroducing
the bias field and then "MPRAGEises" the UNI image.

Usage:
    MPRAGEise.py -i INV2_image -u UNI_image [-r re_bias] [-o output_folder] [-v]

Examples:
    MPRAGEise.py -i /path/to/data/inv2.nii.gz -u /data/uni.nii.gz
    MPRAGEise.py -i /path/to/data/inv2+orig -u /path/to/data/uni+orig -r 1

Created by: Sriranga Kashyap (01-2025), srikashmri@gmail.com
"""

import argparse
import datetime
import json
import logging
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

__version__ = "1.9.2"

# Set AFNI environment variables
os.environ["AFNI_NIFTI_TYPE_WARN"] = "NO"
os.environ["AFNI_ENVIRON_WARNINGS"] = "NO"

log = logging.getLogger("MPRAGEise")

# Common install locations checked after PATH, before giving up.
# ~/abin and /opt/afni are AFNI's own documented install locations (the
# traditional per-user @update.afni.binaries setup, and the system-wide one).
AFNI_FALLBACK_DIRS = ("/opt/afni-latest", "/opt/afni", os.path.expanduser("~/abin"))

# Caches the resolved AFNI directory next to this script, so only the first
# run on a given machine pays for the PATH/fallback-dir scan.
AFNI_CACHE_PATH = Path(__file__).resolve().parent / ".mprageise_afni_cache.json"


class AfniNotFoundError(RuntimeError):
    """Raised when no AFNI installation can be located."""


class AfniCommandError(RuntimeError):
    """Raised when an AFNI command exits with a non-zero status."""


def _read_afni_cache() -> str | None:
    try:
        return json.loads(AFNI_CACHE_PATH.read_text()).get("afni_bin_dir")
    except (OSError, ValueError, AttributeError):
        return None


def _write_afni_cache(afni_dir: str) -> None:
    try:
        AFNI_CACHE_PATH.write_text(
            json.dumps(
                {
                    "afni_bin_dir": afni_dir,
                    "resolved_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                indent=2,
            )
            + "\n"
        )
    except OSError as e:
        log.debug("Could not write AFNI path cache to %s: %s", AFNI_CACHE_PATH, e)


def resolve_afni_path(explicit_path: str | None) -> None:
    """Make sure 'afni' resolves on PATH, searching in priority order:
    -afni-path flag, $AFNI_HOME, a cached result from a previous run,
    already-on-PATH, then a short list of common install directories.
    Raises AfniNotFoundError if none work."""
    for candidate in (explicit_path, os.environ.get("AFNI_HOME")):
        if candidate and (Path(candidate) / "afni").exists():
            os.environ["PATH"] = f"{candidate}:{os.environ['PATH']}"
            _write_afni_cache(candidate)
            return
        if candidate:
            log.warning("AFNI not found in %s", candidate)

    cached_dir = _read_afni_cache()
    if cached_dir and (Path(cached_dir) / "afni").exists():
        os.environ["PATH"] = f"{cached_dir}:{os.environ['PATH']}"
        return

    which_path = shutil.which("afni")
    if which_path:
        _write_afni_cache(str(Path(which_path).parent))
        return

    for candidate in AFNI_FALLBACK_DIRS:
        if (Path(candidate) / "afni").exists():
            os.environ["PATH"] = f"{candidate}:{os.environ['PATH']}"
            _write_afni_cache(candidate)
            return

    raise AfniNotFoundError(
        "Could not locate AFNI. Checked: -afni-path, $AFNI_HOME, cached path, PATH, "
        + ", ".join(AFNI_FALLBACK_DIRS)
        + ". Install AFNI, or pass -afni-path /path/to/afni/bin."
    )


def get_afni_version() -> str:
    """Return the AFNI version string by calling 'afni -ver'."""
    try:
        result = subprocess.run(
            ["afni", "-ver"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown"


def parse_arguments() -> argparse.Namespace:
    # ANSI escape codes for italics; may not be supported in all terminals.
    italic_start = "\033[3m"
    italic_end = "\033[0m"
    epilog_text = (
        f"{italic_start}Nota bene:{italic_end}\n"
        f"   {italic_start}1. By default, output goes to a new timestamped folder next to the INV2 image.{italic_end}\n"
        f"   {italic_start}2. If you're unsure why one would need the re_bias option, you probably don't need it.{italic_end}\n"
        f"   {italic_start}3. Do not use this script for the MP2RAGE T1 map.{italic_end}\n"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Background denoise MP2RAGE UNI images (MPRAGEising) whilst either "
            "removing or reintroducing the bias field."
        ),
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--inv2", required=True, help="MP2RAGE INV2 image (e.g. /path/to/inv2.nii.gz or file+orig)"
    )
    parser.add_argument(
        "-u", "--uni", required=True, help="MP2RAGE UNI image (e.g. /path/to/uni.nii.gz or uni+orig)"
    )
    parser.add_argument(
        "-r", "--re_bias", default="0", help="Reintroduce bias-field (default=0, optional)."
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output folder for processed files (default: a new <UTC timestamp>_MPRAGEise_run "
             "folder next to the INV2 image). If an existing, non-empty folder is given, "
             "-overwrite is enabled automatically.",
    )
    parser.add_argument(
        "-overwrite", action="store_true", default=False, help="Include -overwrite flag in each AFNI command."
    )
    parser.add_argument(
        "-afni-path", default=None, help="Directory containing the AFNI binaries (checked before $AFNI_HOME and PATH)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Enable verbose logging of command execution and debug information.",
    )
    parser.add_argument(
        "-qc", action="store_true", default=False,
        help="Write a QC PNG comparing the UNI image before/after, with normalised intensity "
             "histograms. Requires nibabel and matplotlib: pip install mprageise[full]",
    )
    parser.add_argument("-version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


@dataclass(frozen=True)
class AfniDataset:
    """An MP2RAGE image's identity: its basename and the extension that
    marks its format (.nii, .nii.gz, or +orig for AFNI datasets)."""

    basename: str
    ext: str

    def named(self, suffix: str) -> str:
        return f"{self.basename}_{suffix}{self.ext}"

    @classmethod
    def from_path(cls, filename: str) -> "AfniDataset":
        """
        Determine the base name and file extension.

        For NIfTI files (containing ".nii"):
          - If the filename ends with '.nii.gz', remove that and set ext = '.nii.gz'
          - Otherwise, remove '.nii' and set ext = '.nii'

        For AFNI datasets (if filename contains a '+'):
          - Try to call '@GetAfniPrefix' if available; if not, take the substring before '+'.
          - Set ext = '+orig'
        """
        name = Path(filename).name

        if ".nii" in name:
            if name.endswith(".nii.gz"):
                return cls(name[: -len(".nii.gz")], ".nii.gz")
            return cls(name[: -len(".nii")], ".nii")

        if "+" in name:
            try:
                result = subprocess.run(
                    ["@GetAfniPrefix", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                basename = result.stdout.strip() if result.returncode == 0 else name.split("+")[0]
            except Exception:
                basename = name.split("+")[0]
            return cls(basename, "+orig")

        return cls(name, "")


_STEP_LABELS = {
    "3dUnifize": "Removing bias-field",
    "3dcopy": "Copying dataset",
    "3dinfo": "Reading intensity range",
    "3dcalc": "Running AFNI calc",
}


class _Ticker:
    """Prints a live elapsed-time indicator on one line while a step runs,
    in place of AFNI's own command output. Suppressed when verbose logging
    is on, since the debug log already gives per-command feedback."""

    def __init__(self, label: str):
        self.label = label
        self.ok = True
        self._active = not log.isEnabledFor(logging.DEBUG)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def __enter__(self) -> "_Ticker":
        self._start = time.monotonic()
        if self._active:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        return self

    def _tick(self) -> None:
        while not self._stop.wait(1):
            elapsed = time.monotonic() - self._start
            sys.stdout.write(f"\r  {self.label}... ({elapsed:0.0f}s)")
            sys.stdout.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._active:
            return
        self._stop.set()
        self._thread.join()
        elapsed = time.monotonic() - self._start
        status = "done" if (self.ok and exc_type is None) else "failed"
        sys.stdout.write(f"\r  {self.label}... {status} ({elapsed:0.1f}s)\n")
        sys.stdout.flush()


def run_command(cmd: list[str], label: str | None = None) -> str:
    """Run an external command, raising AfniCommandError if it fails.
    AFNI's own stdout/stderr never reach the screen; a live elapsed-time
    ticker stands in for it, and the full output is only logged at DEBUG (-v)."""
    label = label or _STEP_LABELS.get(cmd[0], cmd[0])
    log.debug("Running command: %s", " ".join(cmd))
    with _Ticker(label) as ticker:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ticker.ok = result.returncode == 0
    log.debug("Command stdout: %s", result.stdout)
    log.debug("Command stderr: %s", result.stderr)
    if result.returncode != 0:
        log.debug("Error running command: %s", " ".join(cmd))
        raise AfniCommandError(
            f"AFNI command '{cmd[0]}' failed (exit {result.returncode}). Re-run with -v for the full output."
        )
    return result.stdout.strip()


class AfniOps(ABC):
    """The AFNI operations MPRAGEise needs. A real adapter runs AFNI's CLI;
    a fake adapter in tests records calls without needing AFNI installed."""

    @abstractmethod
    def unifize(self, image: str, prefix: str, overwrite: bool = False) -> None: ...

    @abstractmethod
    def copy(self, image: str, prefix: str) -> None: ...

    @abstractmethod
    def voxel_range(self, image: str) -> tuple[str, str]:
        """Return (minimum, maximum) voxel intensity as strings."""

    @abstractmethod
    def normalize(self, image: str, int_min: str, int_max: str, prefix: str, overwrite: bool = False) -> None: ...

    @abstractmethod
    def multiply(self, a: str, b: str, prefix: str, overwrite: bool = False) -> None: ...


class SubprocessAfniOps(AfniOps):
    """Real AfniOps adapter: shells out to AFNI's command-line tools."""

    def unifize(self, image: str, prefix: str, overwrite: bool = False) -> None:
        flag = ["-overwrite"] if overwrite else []
        run_command(["3dUnifize", "-quiet"] + flag + ["-prefix", prefix, image], label="Removing bias-field")

    def copy(self, image: str, prefix: str) -> None:
        run_command(["3dcopy", image, prefix], label="Copying dataset")

    def voxel_range(self, image: str) -> tuple[str, str]:
        int_max = run_command(["3dinfo", "-dmaxus", image], label="Reading intensity range")
        int_min = run_command(["3dinfo", "-dminus", image], label="Reading intensity range")
        return int_min, int_max

    def normalize(self, image: str, int_min: str, int_max: str, prefix: str, overwrite: bool = False) -> None:
        flag = ["-overwrite"] if overwrite else []
        expr = f"( a - {int_min} ) / ( {int_max} - {int_min} )"
        run_command(["3dcalc"] + flag + ["-a", image, "-expr", expr, "-prefix", prefix], label="Normalising intensity")

    def multiply(self, a: str, b: str, prefix: str, overwrite: bool = False) -> None:
        flag = ["-overwrite"] if overwrite else []
        run_command(
            ["3dcalc"] + flag + ["-a", a, "-b", b, "-expr", "a * b", "-prefix", prefix],
            label="MPRAGEising the UNI image",
        )


def file_size_bytes(path: str) -> int | None:
    """Best-effort file size. Handles +orig AFNI datasets, which are stored
    as separate .HEAD/.BRIK(.gz) files rather than one literal path."""
    p = Path(path)
    if p.exists():
        return p.stat().st_size
    total = 0
    found = False
    for suffix in (".HEAD", ".BRIK", ".BRIK.gz"):
        companion = Path(f"{path}{suffix}")
        if companion.exists():
            total += companion.stat().st_size
            found = True
    return total if found else None


def write_summary(
    summary_path: Path,
    inv2_image: str,
    uni_image: str,
    out_name: Path,
    command: str,
    qc_path: Path | None = None,
) -> None:
    """Write a JSON record of this run: inputs, the command invoked,
    MPRAGEise's version, when it ran, and input/output file sizes."""
    summary = {
        "mprageise_version": __version__,
        "run_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        "inputs": {
            "inv2": {"path": str(Path(inv2_image).resolve()), "size_bytes": file_size_bytes(inv2_image)},
            "uni": {"path": str(Path(uni_image).resolve()), "size_bytes": file_size_bytes(uni_image)},
        },
        "output": {"path": str(out_name), "size_bytes": file_size_bytes(str(out_name))},
    }
    if qc_path is not None:
        summary["qc_image"] = {"path": str(qc_path), "size_bytes": file_size_bytes(str(qc_path))}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


class QcDependencyError(RuntimeError):
    """Raised when -qc is requested but nibabel/matplotlib aren't installed."""


def _load_qc_libs():
    """Lazily import the -qc feature's optional dependencies, so the base
    install (cp MPRAGEise.py $HOME/abin, stdlib + AFNI only) never needs them."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import nibabel as nib
        import numpy as np
    except ImportError as e:
        raise QcDependencyError(
            "-qc needs nibabel, matplotlib and numpy, which aren't installed. "
            "Install them with: pip install mprageise[full]"
        ) from e
    return nib, np, plt


def _nib_loadable_path(path: str) -> str:
    """nibabel needs the literal .HEAD file for AFNI +orig datasets."""
    if path.endswith("+orig") and not Path(path).exists():
        head = f"{path}.HEAD"
        if Path(head).exists():
            return head
    return path


# QC figure styling: colours are a validated, adjacent CVD-safe pair (slots
# 2 and 3 of the dataviz skill's reference categorical palette); inks follow
# its chart-chrome roles. See CONTEXT.md for why these specific values.
QC_BEFORE_COLOR = "#eb6834"
QC_AFTER_COLOR = "#1baf7a"
QC_PRIMARY_INK = "#0b0b0b"
QC_SECONDARY_INK = "#52514e"
QC_MUTED_INK = "#898781"


def _otsu_threshold(data, np, bins: int = 256) -> float:
    """Classic Otsu threshold (between-class variance maximisation) on a
    coarse histogram of `data`. Pure numpy, no new dependency."""
    counts, edges = np.histogram(data.ravel(), bins=bins)
    counts = counts.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    weight1 = np.cumsum(counts)
    weight2 = weight1[-1] - weight1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean1 = np.cumsum(counts * centers) / weight1
        mean2 = (np.cumsum((counts * centers)[::-1])[::-1]) / weight2
        variance = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[:-1]) ** 2
    variance = np.nan_to_num(variance)
    return edges[1:-1][np.argmax(variance)]


def _foreground_mask(inv2_data, np):
    """Boolean foreground mask from the raw INV2 image (before bias
    correction) via Otsu thresholding. INV2's brain/background contrast is
    much cleaner than UNI's flat non-zero noise floor, so this is a reliable
    stand-in for a real brain mask without needing segmentation."""
    return inv2_data > _otsu_threshold(inv2_data, np)


def _foreground_values(data, mask, np):
    """Apply the INV2-derived mask to `data`, falling back to a simple
    nonzero filter if shapes don't line up (e.g. inputs on different grids -
    shouldn't happen in the normal pipeline, but never crash QC over it)."""
    if mask is not None and mask.shape == data.shape:
        return data[mask]
    log.debug("QC mask shape mismatch; falling back to nonzero voxels.")
    return data[data != 0]


def _plot_qc_panel(ax_img, ax_hist, data, mask, np, title: str, color: str, bins: int = 200) -> float:
    """One QC panel: the volume's middle axial slice (full FOV, background
    included, since it is useful for spotting acquisition artifacts), plus
    its normalised intensity histogram (outline only, foreground voxels
    only, clipped to the 0.05-99.5th percentile range). Returns the
    histogram's y-limit, so the caller can share a scale across both
    panels."""
    middle_slice = data[:, :, data.shape[2] // 2]
    ax_img.imshow(middle_slice.T, cmap="gray", origin="lower")
    ax_img.set_title(title, fontsize=13, fontweight="semibold", color=QC_PRIMARY_INK)
    ax_img.axis("off")

    foreground = _foreground_values(data, mask, np)
    p_low, p_high = np.percentile(foreground, [0.05, 99.5])
    ax_hist.hist(
        foreground, bins=bins, range=(p_low, p_high), density=True,
        histtype="step", linewidth=2, color=color,
    )
    ax_hist.set_xlim(p_low, p_high)
    for spine in ax_hist.spines.values():
        spine.set_visible(False)
    ax_hist.set_xlabel("Intensity", fontsize=10, color=QC_SECONDARY_INK)
    ax_hist.set_ylabel("Normalised count", fontsize=10, color=QC_SECONDARY_INK)
    ax_hist.tick_params(colors=QC_MUTED_INK, labelsize=9, length=0)
    ax_hist.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax_hist.yaxis.get_offset_text().set_fontsize(9)
    ax_hist.yaxis.get_offset_text().set_color(QC_MUTED_INK)

    return ax_hist.get_ylim()[1]


def generate_qc_png(inv2_image: str, before_image: str, after_image: str, qc_path: Path, bins: int = 200) -> None:
    """Write a QC PNG: before/after middle-slice images, each with its
    normalised intensity histogram (outline, percentile-clipped, shared
    y-scale) below it. Background is excluded from the histograms using a
    foreground mask derived from the raw INV2 image (before bias
    correction), not from before/after themselves."""
    nib, np, plt = _load_qc_libs()
    # Silence matplotlib's "findfont" warnings when Helvetica/Arial aren't
    # installed (the common case on Linux) - the fallback to DejaVu Sans
    # still happens, it's just noisy about it otherwise.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]

    inv2 = nib.load(_nib_loadable_path(inv2_image)).get_fdata()
    before = nib.load(_nib_loadable_path(before_image)).get_fdata()
    after = nib.load(_nib_loadable_path(after_image)).get_fdata()
    mask = _foreground_mask(inv2, np) if inv2.shape == before.shape else None

    # US Letter, portrait (8.5in x 11in). Brain-image panels sit in a row,
    # histogram panels in a row below at 85% of the image size. Each row is
    # its own centred pair with an exact 0.5cm gap between its two panels;
    # the gap between the two rows is larger (1.5cm) so the histograms read
    # as clearly separate from the images, not crowded under them. The whole
    # block is centred vertically on the page.
    page_w, page_h = 8.5, 11.0
    gap_in = 0.5 / 2.54  # 0.5cm, within a row
    row_gap_in = 1.5 / 2.54  # 1.5cm, between the image row and the histogram row
    margin_lr = 0.75  # inches, each side - image panel size is derived from this
    img_in = (page_w - 2 * margin_lr - gap_in) / 2
    hist_in = img_in * 0.85  # histogram panels scaled down 15%

    def _row_lefts(panel_in: float) -> list[float]:
        left = (page_w - (2 * panel_in + gap_in)) / 2
        return [left, left + panel_in + gap_in]

    img_lefts = _row_lefts(img_in)
    hist_lefts = _row_lefts(hist_in)

    total_block_h = img_in + row_gap_in + hist_in
    bottom_margin = (page_h - total_block_h) / 2
    hist_bottom = bottom_margin
    img_bottom = bottom_margin + hist_in + row_gap_in

    def _frac(x_in: float, span: float) -> float:
        return x_in / span

    fig = plt.figure(figsize=(page_w, page_h))
    ax_img_before = fig.add_axes((_frac(img_lefts[0], page_w), _frac(img_bottom, page_h), _frac(img_in, page_w), _frac(img_in, page_h)))
    ax_img_after = fig.add_axes((_frac(img_lefts[1], page_w), _frac(img_bottom, page_h), _frac(img_in, page_w), _frac(img_in, page_h)))
    ax_hist_before = fig.add_axes((_frac(hist_lefts[0], page_w), _frac(hist_bottom, page_h), _frac(hist_in, page_w), _frac(hist_in, page_h)))
    ax_hist_after = fig.add_axes((_frac(hist_lefts[1], page_w), _frac(hist_bottom, page_h), _frac(hist_in, page_w), _frac(hist_in, page_h)))

    y1 = _plot_qc_panel(ax_img_before, ax_hist_before, before, mask, np, "Before", QC_BEFORE_COLOR, bins=bins)
    y2 = _plot_qc_panel(ax_img_after, ax_hist_after, after, mask, np, "After", QC_AFTER_COLOR, bins=bins)
    shared_ylim = max(y1, y2)
    ax_hist_before.set_ylim(0, shared_ylim)
    ax_hist_after.set_ylim(0, shared_ylim)
    # The right column's y-axis reads on its right edge, mirrored from the
    # left column, so the two histograms don't crowd their labels together
    # in the gap between them.
    ax_hist_after.yaxis.tick_right()
    ax_hist_after.yaxis.set_label_position("right")

    fig.text(0.05, 0.975, f"MPRAGEise {__version__}", fontsize=9, color=QC_MUTED_INK, ha="left", va="top")
    fig.text(0.05, 0.955, f"UNI: {Path(before_image).name}", fontsize=14, fontweight="bold", color=QC_PRIMARY_INK, ha="left", va="top")
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    fig.text(0.05, 0.025, timestamp, fontsize=9, color=QC_MUTED_INK, ha="left", va="bottom")

    fig.savefig(qc_path, dpi=150, facecolor="white")
    plt.close(fig)


def default_output_folder(inv2_image: str) -> Path:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(inv2_image).resolve().parent / f"{timestamp}_MPRAGEise_run"


def resolve_output_folder(inv2_image: str, output: str | None) -> tuple[Path, bool]:
    """Resolve the output folder, creating it if needed. Returns
    (folder, auto_overwrite) where auto_overwrite is True if an existing,
    non-empty folder was given, so callers know to enable -overwrite."""
    output_folder = Path(output) if output else default_output_folder(inv2_image)
    auto_overwrite = output_folder.is_dir() and any(output_folder.iterdir())
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"Error: Unable to create output folder '{output_folder}': {e}")
    if not os.access(output_folder, os.W_OK):
        sys.exit(f"Error: Output folder '{output_folder}' is not writable.")
    return output_folder, auto_overwrite


def make_work_dir(output_folder: Path) -> Path:
    """Create a scratch directory for intermediate files, inside the output
    folder rather than the OS temp dir, so it shares its filesystem/mount
    with the final output. Caller is responsible for removing it."""
    work_dir = output_folder / f"tmp_mprageise_{secrets.token_hex(4)}"
    try:
        work_dir.mkdir(parents=True)
    except OSError as e:
        sys.exit(f"Error: Unable to create working folder '{work_dir}': {e}")
    return work_dir


def mpragise(
    inv2_image: str,
    uni_image: str,
    output_folder: Path,
    work_dir: Path,
    re_bias: bool,
    overwrite: bool,
    ops: AfniOps,
) -> Path:
    inv2 = AfniDataset.from_path(inv2_image)
    uni = AfniDataset.from_path(uni_image)

    bfc_prefix = str(work_dir / inv2.named("bfc"))
    if re_bias:
        ops.copy(inv2_image, bfc_prefix)
        out_name = output_folder / uni.named("rebiased_clean")
    else:
        ops.unifize(inv2_image, bfc_prefix, overwrite)
        out_name = output_folder / uni.named("unbiased_clean")

    int_min, int_max = ops.voxel_range(inv2_image)
    intnorm_prefix = str(work_dir / inv2.named("intnorm"))
    ops.normalize(bfc_prefix, int_min, int_max, intnorm_prefix, overwrite)

    ops.multiply(uni_image, intnorm_prefix, str(out_name), overwrite)
    return out_name


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.overwrite:
        log.warning("The -overwrite option is enabled; existing output files may be overwritten.")

    try:
        resolve_afni_path(args.afni_path)
    except AfniNotFoundError as e:
        sys.exit(str(e))

    log.info("------------------------------")
    log.info("  MPRAGEise v%s is running.", __version__)
    log.info("------------------------------")
    log.info("AFNI Version: %s", get_afni_version())
    log.info("Run Date: %s", datetime.datetime.now().strftime("%c"))
    log.info("Input files:")
    log.info("  INV2 image: %s", args.inv2)
    log.info("  UNI image : %s", args.uni)

    output_folder, auto_overwrite = resolve_output_folder(args.inv2, args.output)
    overwrite = args.overwrite or auto_overwrite
    if auto_overwrite and not args.overwrite:
        log.warning("Output folder '%s' already contains files; automatically enabling overwrite.", output_folder)
    log.info("Output folder: %s", output_folder)

    re_bias = args.re_bias.strip() != "0"

    work_dir = make_work_dir(output_folder)
    try:
        out_name = mpragise(args.inv2, args.uni, output_folder, work_dir, re_bias, overwrite, SubprocessAfniOps())
    except AfniCommandError as e:
        sys.exit(str(e))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    qc_path = None
    if args.qc:
        uni_ds = AfniDataset.from_path(args.uni)
        qc_path = output_folder / f"{uni_ds.basename}_qc.png"
        try:
            generate_qc_png(args.inv2, args.uni, str(out_name), qc_path)
        except QcDependencyError as e:
            sys.exit(str(e))

    summary_path = output_folder / "mprageise_summary.json"
    write_summary(summary_path, args.inv2, args.uni, out_name, shlex.join(sys.argv), qc_path)

    log.info("Done.")
    log.info("Output file: %s", out_name)
    if qc_path:
        log.info("QC image: %s", qc_path)
    log.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
