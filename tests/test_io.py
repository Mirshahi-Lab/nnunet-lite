"""NIfTI reading and writing: sources, reader conventions and orientation round trips."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nnunet_lite.io import nifti_bytes, read_image, save_nifti, segmentation_image

# Columns of the affine: voxel axis 0 runs anterior->posterior, axis 1 superior->inferior,
# axis 2 left->right, i.e. neither RAS nor a pure flip of it.
PERMUTED = np.array(
    [[0.0, 0.0, 0.8, -10.0], [-0.9, 0.0, 0.0, 20.0], [0.0, -2.5, 0.0, 30.0], [0.0, 0.0, 0.0, 1.0]]
)
LPS = np.diag([-0.8, -0.9, 2.5, 1.0])


def make_image(affine: np.ndarray, shape: tuple[int, int, int] = (5, 6, 7)) -> nib.Nifti1Image:
    data = np.random.default_rng(0).integers(0, 200, size=shape).astype(np.int16)
    return nib.Nifti1Image(data, affine)


def test_every_source_type_reads_the_same_array(tmp_path: Path) -> None:
    image = make_image(LPS)
    path = tmp_path / "image.nii.gz"
    nib.save(image, str(path))
    compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    sources = [path, str(path), compressed, raw, bytearray(raw), io.BytesIO(compressed), image]
    reference, spacing, _ = read_image([path], "NibabelIOWithReorient")
    assert reference.dtype == np.float32
    assert reference.shape == (1, 7, 6, 5)
    assert spacing == pytest.approx((2.5, 0.9, 0.8))
    for source in sources:
        data, source_spacing, _ = read_image([source], "NibabelIOWithReorient")
        assert np.array_equal(data, reference)
        assert source_spacing == spacing


def test_reader_conventions() -> None:
    image = make_image(LPS)
    voxels = np.asarray(image.dataobj)
    plain, spacing, _ = read_image([image], "NibabelIO")
    assert np.array_equal(plain[0], voxels.transpose(2, 1, 0))
    assert spacing == pytest.approx((2.5, 0.9, 0.8))
    # LPS -> RAS flips the first two file axes, the last two of the (Z, Y, X) array.
    reoriented, spacing, _ = read_image([image], "NibabelIOWithReorient")
    assert np.array_equal(reoriented[0], voxels.transpose(2, 1, 0)[:, ::-1, ::-1])
    assert spacing == pytest.approx((2.5, 0.9, 0.8))

    permuted, spacing, _ = read_image([make_image(PERMUTED)], "NibabelIOWithReorient")
    assert permuted.shape == (1, 6, 5, 7)  # (S, A, R) extents
    assert spacing == pytest.approx((2.5, 0.9, 0.8))


@pytest.mark.parametrize("affine", [LPS, PERMUTED, np.diag([1.0, 1.0, 1.0, 1.0])])
@pytest.mark.parametrize("reader", ["NibabelIO", "NibabelIOWithReorient"])
def test_segmentation_returns_to_the_input_grid(affine: np.ndarray, reader: str) -> None:
    image = make_image(affine)
    data, _, geometry = read_image([image], reader)
    # Using the image itself as the "segmentation" must reproduce the input file.
    restored = segmentation_image(data[0].astype(np.uint8), geometry)
    assert restored.shape == image.shape
    assert np.allclose(restored.affine, image.affine)
    assert np.array_equal(np.asarray(restored.dataobj), np.asarray(image.dataobj))


def test_save_and_bytes(tmp_path: Path) -> None:
    image = make_image(PERMUTED)
    data, _, geometry = read_image([image], "NibabelIOWithReorient")
    restored = segmentation_image(data[0].astype(np.uint8), geometry)
    path = save_nifti(restored, tmp_path / "out" / "seg.nii.gz")
    assert [p.name for p in path.parent.iterdir()] == ["seg.nii.gz"]
    assert np.array_equal(np.asarray(nib.load(str(path)).dataobj), np.asarray(image.dataobj))
    for compress in (True, False):
        content = nifti_bytes(restored, compress=compress)
        assert (content[:2] == b"\x1f\x8b") is compress
        reloaded, _, _ = read_image([content], "NibabelIOWithReorient")
        assert np.array_equal(reloaded, data)
    with pytest.raises(ValueError, match=r"\.nii"):
        save_nifti(restored, tmp_path / "seg.mha")


def test_multiple_channels() -> None:
    image = make_image(LPS)
    data, _, _ = read_image([image, image], "NibabelIO")
    assert data.shape == (2, 7, 6, 5)
    with pytest.raises(ValueError, match="shape"):
        read_image([image, make_image(LPS, (5, 6, 8))], "NibabelIO")
    with pytest.raises(ValueError, match="spacing"):
        read_image([image, make_image(np.diag([-0.8, -0.9, 3.0, 1.0]))], "NibabelIO")


def test_rejects_unreadable_input(tmp_path: Path) -> None:
    image = make_image(LPS)
    with pytest.raises(NotImplementedError, match="SimpleITKIO"):
        read_image([image], "SimpleITKIO")
    with pytest.raises(ValueError, match="not a NIfTI"):
        read_image([b"definitely not an image"], "NibabelIO")
    with pytest.raises(FileNotFoundError):
        read_image([tmp_path / "missing.nii.gz"], "NibabelIO")
    with pytest.raises(ValueError, match="3D"):
        read_image([nib.Nifti1Image(np.zeros((4, 4, 4, 2), dtype=np.int16), LPS)], "NibabelIO")
    with pytest.raises(ValueError, match="non-finite voxel"):
        read_image([nib.Nifti1Image(np.full((4, 4, 4), np.nan, dtype=np.float32), LPS)], "NibabelIO")
