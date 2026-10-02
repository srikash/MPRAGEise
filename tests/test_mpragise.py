import json
import os
from datetime import datetime
from pathlib import Path

import pytest

import MPRAGEise as mp


class FakeAfniOps(mp.AfniOps):
    """Records every call instead of shelling out to AFNI."""

    def __init__(self, voxel_range=("100", "900"), fail_on=None):
        self.calls = []
        self._voxel_range = voxel_range
        self._fail_on = fail_on

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self._fail_on == name:
            raise mp.AfniCommandError(f"fake failure in {name}")

    def unifize(self, image, prefix, overwrite=False):
        self._record("unifize", image=image, prefix=prefix, overwrite=overwrite)

    def copy(self, image, prefix):
        self._record("copy", image=image, prefix=prefix)

    def voxel_range(self, image):
        self._record("voxel_range", image=image)
        return self._voxel_range

    def normalize(self, image, int_min, int_max, prefix, overwrite=False):
        self._record("normalize", image=image, int_min=int_min, int_max=int_max, prefix=prefix, overwrite=overwrite)

    def multiply(self, a, b, prefix, overwrite=False):
        self._record("multiply", a=a, b=b, prefix=prefix, overwrite=overwrite)


def run(ops, re_bias=False, overwrite=False, output="/out", work="/work"):
    return mp.mpragise(
        "/data/inv2.nii.gz", "/data/uni.nii.gz", Path(output), Path(work), re_bias, overwrite, ops
    )


def test_bias_removal_calls_unifize_not_copy():
    ops = FakeAfniOps()
    run(ops, re_bias=False)
    names = [name for name, _ in ops.calls]
    assert "unifize" in names
    assert "copy" not in names


def test_rebias_calls_copy_not_unifize():
    ops = FakeAfniOps()
    run(ops, re_bias=True)
    names = [name for name, _ in ops.calls]
    assert "copy" in names
    assert "unifize" not in names


@pytest.mark.parametrize(
    "inv2,uni,ext",
    [
        ("/data/inv2.nii.gz", "/data/uni.nii.gz", ".nii.gz"),
        ("/data/inv2+orig", "/data/uni+orig", "+orig"),
    ],
)
def test_output_filename_patterns(inv2, uni, ext):
    ops = FakeAfniOps()
    out = mp.mpragise(inv2, uni, Path("/out"), Path("/work"), False, False, ops)
    assert out == Path("/out") / f"uni_unbiased_clean{ext}"

    out = mp.mpragise(inv2, uni, Path("/out"), Path("/work"), True, False, ops)
    assert out == Path("/out") / f"uni_rebiased_clean{ext}"


def test_overwrite_flag_threads_through():
    ops = FakeAfniOps()
    run(ops, re_bias=False, overwrite=True)
    for name, kwargs in ops.calls:
        if "overwrite" in kwargs:
            assert kwargs["overwrite"] is True


def test_error_propagates_when_afniops_raises():
    ops = FakeAfniOps(fail_on="unifize")
    with pytest.raises(mp.AfniCommandError):
        run(ops, re_bias=False)


def test_afni_dataset_naming():
    ds = mp.AfniDataset.from_path("/data/sub-01_inv2.nii.gz")
    assert ds.basename == "sub-01_inv2"
    assert ds.ext == ".nii.gz"
    assert ds.named("bfc") == "sub-01_inv2_bfc.nii.gz"


def test_default_output_folder_is_utc_timestamped(tmp_path):
    inv2 = tmp_path / "inv2.nii.gz"
    inv2.touch()
    folder = mp.default_output_folder(str(inv2))
    assert folder.parent == tmp_path
    assert folder.name.endswith("_MPRAGEise_run")
    timestamp = folder.name.removesuffix("_MPRAGEise_run")
    datetime.strptime(timestamp, "%Y%m%d_%H%M%S")  # raises if the format is wrong


def test_resolve_output_folder_creates_fresh_folder_without_auto_overwrite(tmp_path):
    inv2 = tmp_path / "inv2.nii.gz"
    inv2.touch()
    target = tmp_path / "fresh_out"
    folder, auto_overwrite = mp.resolve_output_folder(str(inv2), str(target))
    assert folder == target
    assert folder.is_dir()
    assert auto_overwrite is False


def test_resolve_output_folder_detects_existing_files_and_enables_overwrite(tmp_path):
    inv2 = tmp_path / "inv2.nii.gz"
    inv2.touch()
    target = tmp_path / "existing_out"
    target.mkdir()
    (target / "leftover.nii.gz").touch()
    folder, auto_overwrite = mp.resolve_output_folder(str(inv2), str(target))
    assert folder == target
    assert auto_overwrite is True


def test_resolve_output_folder_empty_existing_folder_no_auto_overwrite(tmp_path):
    inv2 = tmp_path / "inv2.nii.gz"
    inv2.touch()
    target = tmp_path / "empty_out"
    target.mkdir()
    _, auto_overwrite = mp.resolve_output_folder(str(inv2), str(target))
    assert auto_overwrite is False


def test_file_size_bytes_direct_file(tmp_path):
    f = tmp_path / "uni.nii.gz"
    f.write_bytes(b"0123456789")
    assert mp.file_size_bytes(str(f)) == 10


def test_file_size_bytes_afni_dataset_sums_head_and_brik(tmp_path):
    prefix = tmp_path / "uni+orig"
    Path(f"{prefix}.HEAD").write_bytes(b"12345")
    Path(f"{prefix}.BRIK").write_bytes(b"1234567890")
    assert mp.file_size_bytes(str(prefix)) == 15


def test_file_size_bytes_missing_returns_none(tmp_path):
    assert mp.file_size_bytes(str(tmp_path / "nope.nii.gz")) is None


def test_resolve_afni_path_writes_cache_after_resolving(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(mp, "AFNI_CACHE_PATH", cache_path)
    monkeypatch.setattr(mp.shutil, "which", lambda name: "/fake/afni/bin/afni")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("AFNI_HOME", raising=False)

    mp.resolve_afni_path(None)

    cached = json.loads(cache_path.read_text())
    assert cached["afni_bin_dir"] == "/fake/afni/bin"


def test_resolve_afni_path_uses_cache_without_calling_which(monkeypatch, tmp_path):
    afni_dir = tmp_path / "afni_bin"
    afni_dir.mkdir()
    (afni_dir / "afni").touch()
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"afni_bin_dir": str(afni_dir)}))
    monkeypatch.setattr(mp, "AFNI_CACHE_PATH", cache_path)

    def fail_if_called(name):
        raise AssertionError("shutil.which should not be called when the cache is valid")

    monkeypatch.setattr(mp.shutil, "which", fail_if_called)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("AFNI_HOME", raising=False)

    mp.resolve_afni_path(None)

    assert os.environ["PATH"].startswith(str(afni_dir))


def test_resolve_afni_path_raises_with_helpful_message_when_nothing_found(monkeypatch, tmp_path):
    monkeypatch.setattr(mp, "AFNI_CACHE_PATH", tmp_path / "no_cache.json")
    monkeypatch.setattr(mp, "AFNI_FALLBACK_DIRS", (str(tmp_path / "nonexistent-a"), str(tmp_path / "nonexistent-b")))
    monkeypatch.setattr(mp.shutil, "which", lambda name: None)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("AFNI_HOME", raising=False)

    with pytest.raises(mp.AfniNotFoundError) as exc_info:
        mp.resolve_afni_path(None)

    message = str(exc_info.value)
    assert "-afni-path" in message
    assert "$AFNI_HOME" in message
    assert not (tmp_path / "no_cache.json").exists()


def test_write_summary_contents(tmp_path):
    inv2 = tmp_path / "inv2.nii.gz"
    uni = tmp_path / "uni.nii.gz"
    inv2.write_bytes(b"1" * 100)
    uni.write_bytes(b"2" * 200)
    out = tmp_path / "out" / "uni_unbiased_clean.nii.gz"
    out.parent.mkdir()
    out.write_bytes(b"3" * 300)
    summary_path = tmp_path / "mprageise_summary.json"

    mp.write_summary(summary_path, str(inv2), str(uni), out, "python3 MPRAGEise.py -i inv2.nii.gz -u uni.nii.gz")

    summary = json.loads(summary_path.read_text())
    assert summary["mprageise_version"] == mp.__version__
    assert summary["command"] == "python3 MPRAGEise.py -i inv2.nii.gz -u uni.nii.gz"
    assert summary["inputs"]["inv2"]["size_bytes"] == 100
    assert summary["inputs"]["uni"]["size_bytes"] == 200
    assert summary["output"]["size_bytes"] == 300
    datetime.strptime(summary["run_timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ")


def _bimodal_array(rng, shape, low=10, high=200):
    """Synthetic data with two well-separated clusters, half the voxels at
    each - a stand-in for an INV2 image's background/foreground split."""
    import numpy as np

    data = np.where(rng.random(shape) < 0.5, low, high).astype(np.float32)
    return data + rng.normal(0, 2, shape).astype(np.float32)


def test_generate_qc_png_creates_file(tmp_path):
    pytest.importorskip("nibabel")
    pytest.importorskip("matplotlib")
    import numpy as np
    import nibabel as nib

    rng = np.random.default_rng(0)
    shape = (8, 8, 8)
    inv2_data = _bimodal_array(rng, shape)
    before_data = (rng.random(shape) * 100).astype(np.float32)
    after_data = before_data * 2
    affine = np.eye(4)
    inv2 = tmp_path / "inv2.nii.gz"
    before = tmp_path / "before.nii.gz"
    after = tmp_path / "after.nii.gz"
    nib.save(nib.Nifti1Image(inv2_data, affine), inv2)
    nib.save(nib.Nifti1Image(before_data, affine), before)
    nib.save(nib.Nifti1Image(after_data, affine), after)

    qc_path = tmp_path / "qc.png"
    mp.generate_qc_png(str(inv2), str(before), str(after), qc_path, bins=50)

    assert qc_path.exists()
    assert qc_path.stat().st_size > 0


def test_otsu_threshold_separates_bimodal_data():
    pytest.importorskip("numpy")
    import numpy as np

    rng = np.random.default_rng(1)
    data = _bimodal_array(rng, (32, 32, 32), low=10, high=200)
    threshold = mp._otsu_threshold(data, np)
    assert 10 < threshold < 200


def test_foreground_mask_excludes_background():
    pytest.importorskip("numpy")
    import numpy as np

    rng = np.random.default_rng(2)
    data = _bimodal_array(rng, (32, 32, 32), low=10, high=200)
    mask = mp._foreground_mask(data, np)
    assert data[mask].mean() > 150
    assert data[~mask].mean() < 50


def test_load_qc_libs_raises_clear_error_when_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("nibabel", "matplotlib"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(mp.QcDependencyError) as exc_info:
        mp._load_qc_libs()
    assert "mprageise[full]" in str(exc_info.value)
