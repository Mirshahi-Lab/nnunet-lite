from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .bundle import Bundle, load_bundle
from .config import BundleConfig, PreprocessingConfig
from .io import (
    ImageGeometry,
    ImageSource,
    is_image_source,
    load_nifti,
    nifti_bytes,
    read_image,
    save_nifti,
    segmentation_image,
)
from .postprocessing import restore_segmentation
from .preprocessing import PreprocessingProperties, preprocess
from .qc import LabelQC, component_stats, confidence_stats, resolve_labels
from .sliding_window import predict_logits

# Input -> segmentation. Preprocessing always happens here, from the raw input and the
# bundle's own configuration: there is deliberately no way to hand in data that was
# preprocessed elsewhere.

PredictionSource: TypeAlias = "ImageSource | Sequence[ImageSource] | NDArray[Any] | torch.Tensor"
QCSelection: TypeAlias = "bool | Sequence[int | str]"


@dataclass(frozen=True, eq=False)
class PredictionResult:
    # Labels on the CPU, in the axis order of the array the model was given: the input
    # array itself, or for files what the bundle's image reader produced, (Z, Y, X).
    segmentation: torch.Tensor
    labels: dict[int, str]
    qc: dict[str, LabelQC] | None = None
    geometry: ImageGeometry | None = field(default=None, repr=False)

    def _image(self) -> Any:
        if self.geometry is None:
            raise ValueError(
                "This prediction was made from an array, so there is no image geometry "
                "to write a file with; use .segmentation"
            )
        return segmentation_image(self.segmentation.numpy(), self.geometry)

    def save(self, path: str | Path) -> Path:
        return save_nifti(self._image(), path)

    def to_bytes(self, *, compress: bool = True) -> bytes:
        return nifti_bytes(self._image(), compress=compress)


@dataclass(frozen=True, eq=False)
class _Input:
    data: NDArray[Any] | torch.Tensor
    spacing: tuple[float, ...]
    geometry: ImageGeometry | None


def _is_array(source: object) -> bool:
    return isinstance(source, np.ndarray | torch.Tensor)


def _image_sources(source: Any) -> list[Any]:
    # Parsed once, so several models (and readers) can share one in-memory image.
    sources = [source] if is_image_source(source) else list(source)
    if not sources or not all(is_image_source(s) for s in sources):
        raise ValueError(
            "Expected a NIfTI path, bytes, binary file object or nibabel image (or a sequence "
            "of them, one per channel), or an array of shape (C, X, Y, Z) with its spacing"
        )
    return [load_nifti(s) for s in sources]


def _read_input(source: Any, spacing: Sequence[float] | None, config: PreprocessingConfig) -> _Input:
    if _is_array(source):
        if spacing is None:
            raise ValueError("spacing is required when predicting from an array")
        return _Input(source, tuple(float(v) for v in spacing), None)
    if spacing is not None:
        raise ValueError("spacing is read from the image file and cannot be overridden")
    if len(source) != len(config.channels):
        raise ValueError(
            f"Model expects {len(config.channels)} channel image(s) "
            f"({', '.join(c.name for c in config.channels)}), got {len(source)}"
        )
    data, file_spacing, geometry = read_image(source, config.image_reader)
    return _Input(data, file_spacing, geometry)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Predictor:
    def __init__(
        self,
        bundle: Bundle | str | Path,
        *,
        device: str | torch.device | None = None,
        batch_size: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        if isinstance(bundle, Bundle):
            bundle.network.eval().to(self.device)
            self.bundle = bundle
        else:
            self.bundle = load_bundle(bundle, device=self.device)

    @property
    def config(self) -> BundleConfig:
        return self.bundle.config

    def predict(
        self,
        source: PredictionSource,
        spacing: Sequence[float] | None = None,
        *,
        qc: QCSelection = False,
        mirror: bool = False,
    ) -> PredictionResult:
        label_ids = resolve_labels(qc, self.config.labels)
        prepared = source if _is_array(source) else _image_sources(source)
        image = _read_input(prepared, spacing, self.config.preprocessing)
        tensor, properties = preprocess(
            image.data, image.spacing, self.config.preprocessing, device=self.device
        )
        return self._predict_preprocessed(
            tensor, properties, image.geometry, label_ids, qc is not False, mirror
        )

    def _predict_preprocessed(
        self,
        tensor: torch.Tensor,
        properties: PreprocessingProperties,
        geometry: ImageGeometry | None,
        label_ids: tuple[int, ...],
        with_qc: bool,
        mirror: bool,
    ) -> PredictionResult:
        config = self.config
        logits = predict_logits(
            self.bundle.network, tensor, config, mirror=mirror, batch_size=self.batch_size
        )
        confidence = confidence_stats(logits, label_ids) if label_ids else {}
        segmentation = restore_segmentation(logits, properties, config.preprocessing)
        del logits
        qc = None
        if with_qc:
            components = component_stats(segmentation.numpy(), label_ids) if label_ids else {}
            qc = {
                config.labels[label]: LabelQC(**components[label], **confidence[label])
                for label in label_ids
            }
        return PredictionResult(
            segmentation=segmentation, labels=dict(config.labels), qc=qc, geometry=geometry
        )


class MultiPredictor:
    def __init__(
        self,
        bundles: Mapping[str, Bundle | str | Path],
        *,
        device: str | torch.device | None = None,
        batch_size: int = 1,
    ) -> None:
        if not bundles:
            raise ValueError("MultiPredictor needs at least one bundle")
        self.predictors = {
            name: Predictor(bundle, device=device, batch_size=batch_size)
            for name, bundle in bundles.items()
        }

    def predict(
        self,
        source: PredictionSource,
        spacing: Sequence[float] | None = None,
        *,
        qc: QCSelection | Mapping[str, QCSelection] = False,
        mirror: bool = False,
    ) -> dict[str, PredictionResult]:
        selections: dict[str, QCSelection] = {}
        for name in self.predictors:
            selections[name] = qc.get(name, False) if isinstance(qc, Mapping) else qc
        if isinstance(qc, Mapping) and set(qc) - set(self.predictors):
            raise ValueError(
                f"QC requested for unknown bundles: {sorted(set(qc) - set(self.predictors))}"
            )
        label_ids = {
            name: resolve_labels(selections[name], predictor.config.labels)
            for name, predictor in self.predictors.items()
        }

        # Models with the same preprocessing share one preprocessed tensor, and the
        # input is parsed once however many models and readers use it.
        groups: list[tuple[PreprocessingConfig, list[str]]] = []
        for name, predictor in self.predictors.items():
            preprocessing = predictor.config.preprocessing
            for known, names in groups:
                if known == preprocessing:
                    names.append(name)
                    break
            else:
                groups.append((preprocessing, [name]))

        prepared = source if _is_array(source) else _image_sources(source)
        results: dict[str, PredictionResult] = {}
        for preprocessing, names in groups:
            image = _read_input(prepared, spacing, preprocessing)
            device = self.predictors[names[0]].device
            tensor, properties = preprocess(image.data, image.spacing, preprocessing, device=device)
            geometry = image.geometry
            del image
            for name in names:
                results[name] = self.predictors[name]._predict_preprocessed(
                    tensor, properties, geometry, label_ids[name], selections[name] is not False, mirror
                )
            del tensor
        return {name: results[name] for name in self.predictors}
