from __future__ import annotations

import gzip
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from numpy.typing import NDArray

# NIfTI reading and writing that reproduces nnU-Net's nibabel readers, so arrays have
# the axis order and orientation the model was trained on. Sources may be paths or
# in-memory content (bytes, file-like objects), so nothing has to touch the disk.

SUPPORTED_READERS = ("NibabelIO", "NibabelIOWithReorient")
_GZIP_MAGIC = b"\x1f\x8b"

ImageSource: TypeAlias = "str | Path | bytes | bytearray | memoryview | BinaryIO | nib.Nifti1Image"


@dataclass(frozen=True, eq=False)
class ImageGeometry:
    # What is needed to write a segmentation back onto the grid of the input file.
    affine: NDArray[np.float64]
    reader_affine: NDArray[np.float64]
    shape: tuple[int, ...]
    reoriented: bool


def is_image_source(source: object) -> bool:
    return isinstance(source, str | Path | bytes | bytearray | memoryview | nib.Nifti1Image) or (
        hasattr(source, "read") and not hasattr(source, "shape")
    )


def load_nifti(source: ImageSource) -> Any:
    if isinstance(source, nib.Nifti1Image):
        return source
    if isinstance(source, str | Path):
        if not Path(source).is_file():
            raise FileNotFoundError(source)
        return nib.load(str(source))
    content = bytes(source) if isinstance(source, bytes | bytearray | memoryview) else source.read()
    if not isinstance(content, bytes):
        raise ValueError("File-like image sources must be opened in binary mode")
    if content[:2] == _GZIP_MAGIC:
        content = gzip.decompress(content)
    for image_class in (nib.Nifti1Image, nib.Nifti2Image):
        try:
            return image_class.from_bytes(content)
        except Exception:
            continue
    raise ValueError("Image content is not a NIfTI-1 or NIfTI-2 file")


def read_image(
    sources: Sequence[ImageSource], image_reader: str
) -> tuple[NDArray[np.float32], tuple[float, ...], ImageGeometry]:
    # One source per input channel. Returns (C, Z, Y, X) and spacing in the same order,
    # exactly as nnU-Net's reader of that name does.
    if image_reader not in SUPPORTED_READERS:
        raise NotImplementedError(
            f"Reading files for image reader {image_reader!r} is not supported "
            f"(supported: {', '.join(SUPPORTED_READERS)}); pass an array and spacing instead"
        )
    reorient = image_reader == "NibabelIOWithReorient"
    channels = []
    spacings = []
    geometries = []
    for source in sources:
        image = load_nifti(source)
        if image.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI image, got {image.ndim}D")
        affine = np.asarray(image.affine, dtype=np.float64)
        if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError("Input NIfTI has a non-finite or non-invertible affine")
        read = image.as_reoriented(io_orientation(affine)) if reorient else image
        # float64 first, then float32: the same rounding as nnU-Net's reader.
        voxels = read.get_fdata(caching="unchanged").transpose((2, 1, 0))[None]
        if not np.all(np.isfinite(voxels)):
            raise ValueError("Input NIfTI contains non-finite voxel values")
        channels.append(voxels)
        spacings.append(tuple(float(v) for v in read.header.get_zooms()[::-1]))
        geometries.append(
            ImageGeometry(
                affine=affine,
                reader_affine=np.asarray(read.affine, dtype=np.float64),
                shape=tuple(int(v) for v in image.shape),
                reoriented=reorient,
            )
        )
    if any(c.shape != channels[0].shape for c in channels):
        raise ValueError(f"Channel images differ in shape: {[c.shape[1:] for c in channels]}")
    if any(s != spacings[0] for s in spacings):
        raise ValueError(f"Channel images differ in spacing: {spacings}")
    data: NDArray[np.float32] = np.vstack(channels, dtype=np.float32, casting="unsafe")
    return data, spacings[0], geometries[0]


def label_dtype(max_label: int) -> type[np.unsignedinteger[Any]]:
    # Smallest unsigned type that holds every label the model can produce, so all
    # outputs of one model share a type whatever labels a given case contains.
    for dtype in (np.uint8, np.uint16, np.uint32):
        if max_label <= np.iinfo(dtype).max:
            return dtype
    raise ValueError(f"Label {max_label} does not fit an unsigned 32-bit integer")


def segmentation_image(segmentation: NDArray[Any], geometry: ImageGeometry, max_label: int) -> Any:
    # segmentation is (Z, Y, X) as returned by read_image; the result is on the
    # voxel grid of the file it was read from.
    if int(segmentation.max(initial=0)) > max_label or int(segmentation.min(initial=0)) < 0:
        raise ValueError(f"Segmentation has values outside the model's labels 0..{max_label}")
    dtype = label_dtype(max_label)
    voxels = segmentation.transpose((2, 1, 0)).astype(dtype, copy=False)
    if geometry.reoriented:
        canonical = nib.Nifti1Image(voxels, geometry.reader_affine)
        image = canonical.as_reoriented(
            ornt_transform(axcodes2ornt("RAS"), io_orientation(geometry.affine))
        )
    else:
        image = nib.Nifti1Image(voxels, geometry.affine)
    if tuple(image.shape) != geometry.shape:
        raise RuntimeError(f"Restored shape {image.shape} does not match the input {geometry.shape}")
    if not np.allclose(image.affine, geometry.affine, atol=1e-5):
        raise RuntimeError("Restored affine does not match the input affine")
    return image


def save_nifti(image: Any, path: str | Path) -> Path:
    output = Path(path)
    if not output.name.endswith((".nii", ".nii.gz")):
        raise ValueError(f"Output file must end in .nii or .nii.gz: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".nii.gz" if output.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=suffix, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        nib.save(image, str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def nifti_bytes(image: Any, *, compress: bool = True) -> bytes:
    content: bytes = image.to_bytes()
    return gzip.compress(content, compresslevel=1) if compress else content
