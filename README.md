# nnunet-lite

Run [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) models without nnU-Net.

`nnunet-lite` converts a trained nnU-Net model into a small, self-contained **bundle** (`bundle.json` + `weights.safetensors`) and provides an inference API that reproduces nnU-Net's preprocessing, sliding-window prediction and resampling. Segmentations match nnU-Net within a small tolerance, with no `nnunetv2` dependency at inference time.

It is a library for conversion and inference, not a serving platform: it makes no attempt at cloud deployment, queuing or orchestration.

## Why

- **Independent of nnU-Net.** A bundle needs only PyTorch, NumPy, SciPy, nibabel and safetensors.
- **Tensor in, tensor out.** The network is a plain `torch.nn.Module`. Predictions accept paths, bytes, file objects, NIfTI images or raw arrays, so nothing has to touch the disk.
- **Safe to load.** Weights are stored as safetensors and checked against a SHA-256 recorded in `bundle.json`. Conversion reads nnU-Net checkpoints with `weights_only=True` and refuses anything else in the pickle.
- **Fast preprocessing.** An optional GPU path is roughly 12–40x faster than nnU-Net's CPU preprocessing, with an exact CPU path when you need bit-equality.

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/). Python 3.11+ is required.

```bash
git clone <repo-url> && cd nnunet_lite
uv sync
```

## Quick start

```python
import nnunet_lite as nl

# 1. Convert once. Point at the nnU-Net trainer__plans__configuration folder.
nl.convert(
    "nnUNet_results/Dataset485_x/nnUNetTrainer__nnUNetPlans__3d_fullres",
    output_dir="bundles/dataset485",
)

# 2. Predict anywhere, with no nnU-Net files or dependencies.
predictor = nl.Predictor("bundles/dataset485")  # CUDA if available
result = predictor.predict("ct.nii.gz", qc=True)

result.segmentation  # CPU uint8 tensor on the input's grid
result.qc["muscle"].estimated_dice  # optional per-label quality control
result.save("seg.nii.gz")
```

## Capabilities

### Conversion

```python
bundle = nl.convert(model_dir)  # convert in memory
bundle = nl.convert(model_dir, output_dir="bundles/x")  # convert and save
bundle.save("bundles/x")  # or save later
bundle = nl.load_bundle("bundles/x", device="cuda")  # load, verifying weights
config = nl.load_config("bundles/x")  # bundle.json only, no weights
```

`convert` takes `fold`, `checkpoint`, `overwrite`, `device`, and an `inference` override for the sliding-window settings (default: nnU-Net's own).

### Preprocessing

`nl.preprocess(volume, spacing, config.preprocessing)` runs nnU-Net's pipeline: transpose, crop to nonzero, normalize, resample. It needs only the config, not the weights.

| | CPU (`device="cpu"`) | GPU (default when CUDA is available) |
|---|---|---|
| Match to nnU-Net | Byte-equal to `DefaultPreprocessor.run_case_npy` | Within float32 rounding |
| Max difference from CPU path | — | ~5e-7 on normalized data |
| Voxels bit-identical to CPU | 100% | ~93% |
| Crop box and shapes | nnU-Net's | Identical to CPU |

The GPU path applies spline resampling as one float64 matrix product per axis, using matrices from the same scipy call the CPU path makes. If it fails for any reason (out of memory, CUDA error, non-finite input) it emits a `RuntimeWarning` and falls back to the exact CPU path.

Measured on an RTX 2080 Ti, resampling to 1.5 mm:

| Input | CPU | GPU | Speedup | Peak GPU memory |
|---|---|---|---|---|
| 300×512×512 at 0.8 mm | 8.76 s | 0.30 s | 29x | 1.3 GB |
| 80×512×512, 5 mm slices | 2.32 s | 0.11 s | 22x | 0.6 GB |
| 600×512×512 at 1×0.7 mm | 17.68 s | 0.45 s | 39x | 3.6 GB |
| 2-channel MRI 150×240×240, masked z-score | 2.82 s | 0.24 s | 12x | 0.3 GB |

Use the default for throughput; use `device="cpu"` when you need exactly what nnU-Net would produce, for example when validating a conversion. The effect of GPU rounding on final segmentations has not been measured yet.

### Prediction

```python
predictor.predict("ct.nii.gz")  # path
predictor.predict(s3_body_bytes)  # bytes or binary file object, nothing written to disk
predictor.predict(volume, spacing)  # raw (C, X, Y, Z) array or tensor plus spacing
predictor.predict(source, qc=["muscle"], mirror=True)
```

- Preprocessing always runs inside `predict` using the bundle's own configuration. There is no entry point for pre-preprocessed tensors.
- **Sliding window** is a port of `nnUNetPredictor`: tile positions, Gaussian weighting, zero padding, float16 accumulation and CUDA autocast. On CUDA out-of-memory it warns and moves accumulation to the CPU. 2D bundles run slice by slice.
- **Restoring** resamples logits to the cropped input shape, argmaxes, un-crops and transposes back, one class at a time so full-resolution logits for all classes are never held at once.
- **Mirroring** (test-time augmentation) is **off** by default, unlike nnU-Net. Pass `mirror=True` to enable it.
- **Output**: `result.save(path)` and `result.to_bytes()` write NIfTI on the input file's own grid and orientation, using the smallest unsigned integer type that holds the model's highest label. These need a file input, since array inputs carry no affine.
- **Inputs**: `.nii` and `.nii.gz` are both accepted, including from memory, via nnU-Net's `NibabelIO` and `NibabelIOWithReorient` readers. Other readers work for arrays only.

### Quality control

`qc=True` (every foreground label) or a list of label names or ids adds per-label statistics. QC never changes the segmentation and costs nothing unless requested.

- **Segmentation stats** (26-connected components): `voxel_count`, `component_count`, `largest_component_fraction`, `small_component_voxel_counts`, `touches_boundary`.
- **Confidence stats** (softmax at network resolution): `estimated_dice`, `mean_margin`, `low_margin_fraction`, `mean_entropy`, `network_voxel_count`.

`estimated_dice` is **uncalibrated**: it is the raw confidence-based estimate at temperature 1. Per-model temperatures and calibration against true Dice are not implemented yet, so treat it as a relative signal, not a Dice value.

### Multiple models on one input

```python
multi = nl.MultiPredictor({"299": "bundles/dataset299", "485": "bundles/dataset485"})
results = multi.predict("ct.nii.gz", qc=True)  # dict[str, PredictionResult]
```

The input is parsed once and preprocessed once per distinct preprocessing config. All networks stay on the device; models run one after another.

## Constraints

**Models**
- Architectures: `PlainConvUNet` and `ResidualEncoderUNet`, 2D and 3D, with InstanceNorm, LeakyReLU or ReLU, and no dropout. Legacy and `architecture`-style plans, including `inherits_from`, are supported.
- One fold per bundle. No ensembling across folds.
- Not supported (raises `NotImplementedError`): cascade configurations, region-based labels, the ignore label.

**Preprocessing**
- Normalization: `CTNormalization`, `ZScoreNormalization` (with and without mask), `RescaleTo01Normalization`, `RGBTo01Normalization`, `NoNormalization`.
- Resampling: only nnU-Net's default, `resample_data_or_seg_to_shape`. Others raise `NotImplementedError`.
- Array inputs must use the axis and spacing order nnU-Net's reader produced for the model, recorded as `config.preprocessing.image_reader`. For example, the nibabel readers give `(z, y, x)` with reversed spacing. 2D images need a singleton first spatial axis, as in nnU-Net.
- Masked z-score still computes its hole-filled mask on the CPU, even on the GPU path.

**Scope**
- The target is nnU-Net-quality output within a low tolerance, not bitwise-identical segmentations. Network logits are tested bitwise-equal to nnU-Net's own network on CPU FP32 and CUDA autocast.
- Bundles record the sliding-window defaults (`tile_step_size=0.5`, Gaussian weighting) because nnU-Net keeps them outside its plans. Override them at conversion time if you need different values.
- Schema 1 bundles still load; saving always writes schema 2.

## Development

```bash
uv run pytest            # synthetic nnU-Net folders, no nnunetv2 needed
uv run ruff check . && uv run ruff format .
uv run mypy
```

Equivalence tests compare against nnU-Net itself and need `nnunetv2`, so they run in the leadup environment:

```bash
PYTHONPATH=src packages/nnunet_model_conversion/.venv/bin/python -m pytest tests
```

- `tests/test_nnunet_equivalence.py`: bitwise-equal logits (needs real model folders).
- `tests/test_nnunet_preprocessing_equivalence.py`: byte-equal preprocessing against `DefaultPreprocessor`.
- `tests/test_nnunet_prediction_equivalence.py`: whole-CT segmentations against `nnUNetPredictor` (needs CUDA and a test CT via `NNUNET_LITE_TEST_IMAGE`).

The earlier speed, accuracy and precision work behind this library lives in `packages/nnunet_model_conversion`.
