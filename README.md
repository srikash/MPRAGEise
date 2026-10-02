# MPRAGEise [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14926753.svg)](https://zenodo.org/badge/latestdoi/14926753)

Background denoise MP2RAGE UNI images (MPRAGEising) whilst either removing or reintroducing the bias field - Powered by **AFNI**

## Table of Contents

- [Usage](#usage)
- [Development](#development)
- [Containerised usage](#containerised-usage)
- [Expected result](#expected-result)

## Usage

**Preferred script:** [MPRAGEise.py](https://github.com/srikash/3dMPRAGEise/blob/main/MPRAGEise.py)  
*MPRAGEise.py is the Python implementation of the original shell script and is recommended for its easier interface and better logging capabilities.*

**Setup:**

`git clone https://github.com/srikash/3dMPRAGEise.git`  
`cp 3dMPRAGEise/MPRAGEise.py $HOME/abin`  

This install needs nothing beyond AFNI and the Python standard library. For `-qc` (below), either install its extra dependencies yourself (`pip install nibabel matplotlib`), or install the package instead: `pip install mprageise[full]`.

**Usage:**

`python MPRAGEise.py -i INV2_image.nii.gz -u UNI_image.nii.gz`

**Options:**

| Flag | Description |
| --- | --- |
| `-i`, `--inv2` | MP2RAGE INV2 image (required) |
| `-u`, `--uni` | MP2RAGE UNI image (required) |
| `-r`, `--re_bias` | Reintroduce bias-field instead of removing it (default `0`) |
| `-o`, `--output` | Output folder (default: a new `<UTC timestamp>_MPRAGEise_run` folder next to the INV2 image) |
| `-overwrite` | Pass `-overwrite` to every AFNI command |
| `-afni-path` | Directory containing the AFNI binaries, checked before `$AFNI_HOME` and `PATH` |
| `-v`, `--verbose` | Verbose logging |
| `-qc` | Write a before/after QC PNG with normalised intensity histograms (needs `pip install mprageise[full]`) |
| `-version` | Print the version and exit |

AFNI is located automatically, in order: `-afni-path`, `$AFNI_HOME`, a cached result from a previous run (`.mprageise_afni_cache.json`, next to the script; delete it if AFNI moves), `PATH`, and then common install directories (`/opt/afni-latest`, `/opt/afni`, and `~/abin`, AFNI's own documented install locations). If none of those resolve, the script exits with a clear error instead of failing deep inside an AFNI call.

If `-o` names an existing, non-empty folder, `-overwrite` is enabled automatically for that run (with a warning) instead of failing partway through.

AFNI's own command output never reaches the screen. A live elapsed-time indicator shows per-step progress instead; pass `-v` to see the full command, stdout, and stderr for each step.

Intermediate files live briefly in a `tmp_mprageise_<8-char hex>` subfolder of the output folder, not the OS temp directory, so they share its filesystem. The subfolder is removed automatically when the run finishes or fails.

`-qc` writes `<uni_basename>_qc.png` into the output folder: a US Letter page with a centred, borderless 2x2 grid showing the UNI image before and after MPRAGEising (middle axial slice) with its normalised intensity histogram below (200 bins, clipped to the 0.05-99.5th percentile range, outline style in validated colours, y-axis in scientific notation). The two images are an exact 0.5cm apart, as are the two histograms (85% of the image size, with the "after" histogram's y-axis mirrored to its right edge); the histogram row sits a clear 1.5cm below the image row. The header names the MPRAGEise version and the UNI input file; the footer gives the UTC timestamp. Background is excluded from the histograms using a foreground mask derived from the raw INV2 image, before bias correction, via Otsu thresholding, since INV2's brain/background contrast is much cleaner than UNI's flat non-zero noise floor would otherwise allow.


*Nota bene: Do not use this script for the MP2RAGE T1 map.*

## Development

`pip install pytest && pytest tests/`

Tests run against a fake AFNI adapter and need no AFNI install. CI runs them on every push/PR via `.github/workflows/tests.yml`.

## Containerised usage

A `Dockerfile` is provided (built with [neurodocker](https://github.com/ReproNim/neurodocker): Fedora 40 + AFNI binaries + Python 3.12), so no local AFNI install is required.

**Build:**

`docker build -t mpragise .`

**Run:**

`docker run --rm -u $(id -u):$(id -g) -v /path/to/data:/data mpragise -i /data/inv2.nii.gz -u /data/uni.nii.gz -o /data/out`

*Nota bene: passing `-u $(id -u):$(id -g)` ensures output files in the mounted volume are owned by you, not the container's default user.*


## Expected result
Compared to the output of [MP2RAGE Robust Background Removal MATLAB Script](https://github.com/JosePMarques/MP2RAGE-related-scripts/blob/master/DemoRemoveBackgroundNoise.m) or [MP2RAGE Robust Background Removal Python Script](https://github.com/khanlab/mp2rage_genUniDen/blob/master/mp2rage_genUniDen.py)

![Coronal](img/coronal.png)
![Axial](img/axial.png)



