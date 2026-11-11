# Overview

This is a library for conversion, storage, and inference of a nnUNet model format with a `safetensors` format and a unified `bundle.json` file. This library is explicitly *not* for building out a cloud production environment. It should simply provide the ability to convert an nnUNet model, store it as a simple bundle, and then provide the API to run inference with the model, matching the necessary preprocessing steps reproduce nnUNet quality outputs (within a low margin of tolerance). 

## Goals

The models should be completely independent of nnUNet, allowing for greater flexibility of input options (particular in production use cases). Specifically, we want to focus on the ability to do tensor in -> tensor out

## Tools

Runtime dependencies beyond torch/numpy/scipy/nibabel/safetensors: `connected-components-3d`, `fastremap` (QC component stats) and `fastmorph` (hole filling in preprocessing).

Use `uv` as the package manager of choice

Use `ruff` for formatting and checking

Use `mypy` for type checking

## Leadup Work

in `packages/nnunet_model_conversion` there has been a lot of work for speed, accuracy, and precision testing on conversions. Use this as a jumping off point for this simplified library. 

## Rules

- Do not regenerate the docstrings in the source code. I will handle that myself. 
- Try not to change the public api without consulting me first.
- Document any changes to the public api in this file.

## High Level Plan

This project will have two main components:

1. **Conversion**: A module to convert an nnUNet model to our new format.
2. **Inference**: A module to run inference with the new format.

## Public API

Everything below is exported from `nnunet_lite`. Update this section whenever the public API changes.

```python
import nnunet_lite as nl

# Convert on the fly: point at the nnU-Net trainer__plans__configuration folder.
bundle = nl.convert("nnUNet_results/Dataset485_x/nnUNetTrainer__nnUNetPlans__3d_fullres")
logits = bundle.network(patch)  # plain torch.nn.Module, eval mode
bundle.config  # BundleConfig: network, preprocessing, patch size, labels

# ...and/or save the production bundle (bundle.json + weights.safetensors)
bundle.save("bundles/dataset485")
nl.convert(model_dir, output_dir="bundles/dataset485")  # convert + save in one call

# Later, with no nnU-Net files or dependencies
bundle = nl.load_bundle("bundles/dataset485", device="cuda")

# Preprocessing is independent of prediction: it needs only the config, not the weights.
config = nl.load_config("bundles/dataset485")  # reads bundle.json only
tensor, properties = nl.preprocess(volume, spacing, config.preprocessing)  # GPU if available
tensor, properties = nl.preprocess(
    volume, spacing, config.preprocessing, device="cpu"
)  # exact nnU-Net path

# Prediction: file, in-memory NIfTI or raw array in -> segmentation out. Preprocessing always runs inside.
predictor = nl.Predictor("bundles/dataset485")  # or a Bundle; CUDA if available
result = predictor.predict("ct.nii.gz")  # path
result = predictor.predict(
    s3_body_bytes, qc=True
)  # bytes / binary file object: nothing touches the disk
result = predictor.predict(volume, spacing)  # raw (C, X, Y, Z) array or tensor + spacing
result.segmentation  # CPU tensor, same grid as the array the model was given
result.qc["muscle"].estimated_dice  # opt-in QC, one LabelQC per label
result.save("seg.nii.gz")
payload = result.to_bytes()  # file inputs: back on the input file's grid

# Several bundles on one input: read once, preprocess once per distinct preprocessing config.
multi = nl.MultiPredictor(
    {"299": "bundles/dataset299", "485": "bundles/dataset485", "998": "bundles/dataset998"}
)
results = multi.predict("ct.nii.gz", qc=True)  # dict[str, PredictionResult]
```

- `convert(model_dir, *, fold=0, checkpoint="checkpoint_final.pth", output_dir=None, overwrite=False, device="cpu", inference=None) -> Bundle` — `inference` overrides the sliding-window settings recorded in the bundle (default: nnU-Net's).
- `load_bundle(bundle_dir, *, device="cpu") -> Bundle` — verifies the weights SHA-256 recorded in `bundle.json`.
- `Bundle` — `config: BundleConfig`, `network: torch.nn.Module`, `save(output_dir, *, overwrite=False) -> Path`.
- `load_config(bundle_dir) -> BundleConfig` — reads `bundle.json` only; weights are not loaded or verified.
- `preprocess(data, spacing, config: PreprocessingConfig, *, device=None) -> tuple[torch.Tensor, PreprocessingProperties]` — transpose → crop to nonzero → normalize → resample. `data` is a numpy array or torch tensor of shape `(C, X, Y, Z)` (never modified); `spacing` has one value per spatial axis. `device=None` uses CUDA when available, otherwise CPU. Returns a float32 tensor on the device it was computed on.
- `PreprocessingProperties` — `spacing`, `shape_before_cropping`, `crop_bbox`, `shape_after_cropping`; what is needed to map a prediction back onto the input array.
- `BundleConfig`, `NetworkConfig`, `PreprocessingConfig`, `ChannelConfig`, `ResamplingConfig`, `InferenceConfig` — frozen dataclasses mirroring `bundle.json` (`schema_version` 2).
- `InferenceConfig` — `tile_step_size=0.5`, `use_gaussian=True`, `gaussian_sigma_scale=0.125`, `gaussian_value_scaling_factor=10.0`; stored as `bundle.json["inference"]` / `BundleConfig.inference`. nnU-Net keeps these as predictor defaults rather than in the plans, so conversion records those defaults unless `convert(..., inference=...)` says otherwise. Schema 1 bundles (no `inference` section) still load, with the defaults; saving always writes schema 2.
- `Predictor(bundle, *, device=None, batch_size=1, compile=False)` — `bundle` is a `Bundle` (its network is moved to `device`) or a bundle directory. `device=None` uses CUDA when available. `batch_size` is the number of sliding-window patches per forward pass. `compile=True` runs the network through `torch.compile` (default mode); see Prediction scope.
- `Predictor.predict(source, spacing=None, *, qc=False, mirror=False) -> PredictionResult`
  - `source`: a NIfTI path, `bytes`, binary file object or `nibabel.Nifti1Image` (or a sequence of them, one per input channel); or a raw numpy array / torch tensor of shape `(C, X, Y, Z)` in the model reader's axis order, with `spacing` (required for arrays, rejected for files).
  - `qc`: `False`, `True` (every foreground label), or a sequence of label names / ids.
  - `mirror`: test-time mirroring over `config.mirror_axes`. Off by default (nnU-Net's own default is on).
- `PredictionResult` — `segmentation` (CPU tensor, uint8, or int16 above 255 labels), `labels`, `qc: dict[str, LabelQC] | None`, `save(path) -> Path`, `to_bytes(*, compress=True) -> bytes`. `save`/`to_bytes` need a file input.
- `LabelQC` — per label: `voxel_count`, `component_count`, `largest_component_fraction`, `small_component_voxel_counts`, `touches_boundary` (26-connected components of the output segmentation); `estimated_dice`, `mean_margin`, `low_margin_fraction`, `mean_entropy`, `network_voxel_count` (softmax confidence at network resolution).
- `MultiPredictor(bundles: Mapping[str, Bundle | path], *, device=None, batch_size=1, compile=False)` and `MultiPredictor.predict(source, spacing=None, *, qc=False, mirror=False) -> dict[str, PredictionResult]` — `qc` may also be a mapping of bundle name to a selection.

### Conversion scope

- Architectures: `PlainConvUNet` and `ResidualEncoderUNet`, 2D and 3D, InstanceNorm, LeakyReLU/ReLU, no dropout. Legacy and `architecture`-style plans, including `inherits_from`.
- One fold per bundle. Not supported (raises `NotImplementedError`): cascade configurations, region-based labels, ignore label.
- Checkpoints are loaded with `weights_only=True` plus a numpy-scalar allowlist; anything else in the pickle is refused rather than executed.

### Preprocessing scope

- `device="cpu"`: byte-equal to nnU-Net's `DefaultPreprocessor.run_case_npy` for the same array and spacing. Runs in numpy/scipy (float64 resampling, as nnU-Net does).
- GPU (a deliberate change from nnU-Net, torch only, no extra dependency): the same steps in torch. Spline resampling is applied as one float64 matrix product per axis, with the matrices taken from the same scipy call the CPU path makes, so output differs from the CPU path only by float32 rounding (max abs difference ~5e-7 on normalized data, ~93% of voxels bit-identical) and crop box/shapes are identical. Roughly 20–40x faster than CPU on full-size CTs (RTX 2080 Ti). Only the hole-filled mask for masked z-score still runs on CPU.
- Hole filling (both paths) uses `fastmorph.fill_holes` on the cropped nonzero mask, and only when a channel is normalized inside the mask; this gives the same mask as nnU-Net's `binary_fill_holes` on the whole volume followed by the crop.
- If the GPU path fails for any reason (out of memory, CUDA error, non-finite input) a `RuntimeWarning` is emitted and the exact CPU path runs instead; the result is then a CPU tensor.
- The input array must use the axis order and spacing order nnU-Net's image reader produced for the model. The reader's name is recorded as `config.preprocessing.image_reader` (e.g. the nibabel readers return `(z, y, x)` with spacing reversed, `NibabelIOWithReorient` after reorienting to RAS). `preprocess` itself takes arrays only; `Predictor` reads files and reorients them (see Prediction scope).
- Normalization: `CTNormalization`, `ZScoreNormalization` (with and without mask), `RescaleTo01Normalization`, `RGBTo01Normalization`, `NoNormalization`.
- Resampling: `resample_data_or_seg_to_shape` only (nnU-Net's default); others raise `NotImplementedError`.
- 2D images need a singleton first spatial axis, as in nnU-Net.

### Prediction scope

- Preprocessing is not optional and not pluggable: `predict` always runs `preprocess` on the raw input with the bundle's own configuration. There is no entry point for already-preprocessed tensors.
- File input (`nnunet_lite.io`, nibabel): reproduces nnU-Net's `NibabelIO` and `NibabelIOWithReorient` readers, chosen by `config.preprocessing.image_reader`; other readers raise `NotImplementedError` for files (arrays still work). `.nii` and gzip content are both accepted from memory. `save`/`to_bytes` write a NIfTI-1 image on the input file's own voxel grid and orientation and check the affine. Its data type is the smallest unsigned integer that holds the model's highest label (`uint8` up to 255, then `uint16`, `uint32`), chosen from the bundle's labels rather than the labels present, so every output of a model has the same type.
- Sliding window: port of `nnUNetPredictor` (tile positions, Gaussian weights, zero padding, float16 accumulation, autocast on CUDA), driven by `config.patch_size`, `config.mirror_axes` and `config.inference`. Accumulates on the GPU; on CUDA out-of-memory a `RuntimeWarning` is emitted and accumulation moves to the CPU. 2D bundles are run slice by slice.
- `compile=True` (opt-in, a deliberate change from nnU-Net, meant for a fixed Linux/CUDA setup): the sliding window calls `torch.compile(network)`; `bundle.network` itself stays the plain module. Measured on an RTX 2080 Ti with two bundles (PlainConvUNet 5 classes, ResidualEncoderUNet 67 classes) and two CTs: sliding window 18–20% faster, about 15% end to end. Logits are no longer bit-equal to nnU-Net's; segmentations differed from the uncompiled ones in a few hundred of 20–52M voxels (lowest per-label Dice 0.9995). The first prediction pays the compilation (about 30 s per model), and a different patch batch shape (the last batch when `batch_size > 1` does not divide the tile count) compiles again. The other `torch.compile` modes were no faster; `max-autotune` took 7–16 minutes to compile. The default (`compile=False`) is unchanged and stays bit-equal.
- Restoring: logits are resampled to the cropped input shape with the bundle's `resampling_probabilities` settings, argmaxed, un-cropped and transposed back, as nnU-Net does. Classes are resampled one at a time against a running maximum, so full-resolution logits of all classes are never held at once. Runs on the predictor's device with float32 matrix products (the result is rounded to float16 before the argmax, as in nnU-Net; float64 gave the same segmentation voxel for voxel and is ~15x slower on consumer GPUs). Logits that the sliding window had to keep on the CPU are still resampled on the GPU, one class at a time. A GPU failure warns and falls back to the numpy path.
- QC never changes the segmentation and costs nothing unless requested. Component stats are computed per label (components never span labels; unselected labels are skipped) with `cc3d` for the components and `fastremap` for the label bookkeeping. Confidence stats come from `packages/nnunet_model_conversion/runtime/scripts/qc_confidence.py`: `estimated_dice = 2·Σ_{pred=k} p_k / (Σ p_k + |pred=k|)` at temperature 1. Not included yet: per-model temperatures and the calibration of estimated Dice to true Dice (with its reject/review thresholds), so `estimated_dice` is uncalibrated.
- `MultiPredictor` keeps every network on the device, parses the input once, decodes the voxel data once per image reader, preprocesses once per distinct `PreprocessingConfig`, and runs the models one after another.

## Testing

- `uv run pytest` — synthetic nnU-Net folders, no nnunetv2 needed.
- `tests/test_nnunet_equivalence.py` checks bitwise-equal logits against nnU-Net's own network (CPU FP32 and CUDA autocast), `tests/test_nnunet_preprocessing_equivalence.py` checks byte-equal preprocessing against nnU-Net's `DefaultPreprocessor`, and `tests/test_nnunet_prediction_equivalence.py` compares whole-CT segmentations with `nnUNetPredictor` (needs CUDA and a test CT, `NNUNET_LITE_TEST_IMAGE`). All need nnunetv2 (the first and last also real model folders), so run them with the leadup environment:
  `PYTHONPATH=src packages/nnunet_model_conversion/.venv/bin/python -m pytest tests`
