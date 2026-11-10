# Overview

This is a library for conversion, storage, and inference of a nnUNet model format with a `safetensors` format and a unified `bundle.json` file. This library is explicitly *not* for building out a cloud production environment. It should simply provide the ability to convert an nnUNet model, store it as a simple bundle, and then provide the API to run inference with the model, matching the necessary preprocessing steps reproduce nnUNet quality outputs (within a low margin of tolerance). 

## Goals

The models should be completely independent of nnUNet, allowing for greater flexibility of input options (particular in production use cases). Specifically, we want to focus on the ability to do tensor in -> tensor out

## Tools

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
logits = bundle.network(patch)      # plain torch.nn.Module, eval mode
bundle.config                       # BundleConfig: network, preprocessing, patch size, labels

# ...and/or save the production bundle (bundle.json + weights.safetensors)
bundle.save("bundles/dataset485")
nl.convert(model_dir, output_dir="bundles/dataset485")   # convert + save in one call

# Later, with no nnU-Net files or dependencies
bundle = nl.load_bundle("bundles/dataset485", device="cuda")

# Preprocessing is independent of prediction: it needs only the config, not the weights.
config = nl.load_config("bundles/dataset485")            # reads bundle.json only
tensor, properties = nl.preprocess(volume, spacing, config.preprocessing)
```

- `convert(model_dir, *, fold=0, checkpoint="checkpoint_final.pth", output_dir=None, overwrite=False, device="cpu") -> Bundle`
- `load_bundle(bundle_dir, *, device="cpu") -> Bundle` — verifies the weights SHA-256 recorded in `bundle.json`.
- `Bundle` — `config: BundleConfig`, `network: torch.nn.Module`, `save(output_dir, *, overwrite=False) -> Path`.
- `load_config(bundle_dir) -> BundleConfig` — reads `bundle.json` only; weights are not loaded or verified.
- `preprocess(data, spacing, config: PreprocessingConfig) -> tuple[torch.Tensor, PreprocessingProperties]` — transpose → crop to nonzero → normalize → resample. `data` is a numpy array or torch tensor of shape `(C, X, Y, Z)` (never modified); `spacing` has one value per spatial axis. Returns a float32 CPU tensor.
- `PreprocessingProperties` — `spacing`, `shape_before_cropping`, `crop_bbox`, `shape_after_cropping`; what is needed to map a prediction back onto the input array.
- `BundleConfig`, `NetworkConfig`, `PreprocessingConfig`, `ChannelConfig`, `ResamplingConfig` — frozen dataclasses mirroring `bundle.json` (`schema_version` 1).

### Conversion scope

- Architectures: `PlainConvUNet` and `ResidualEncoderUNet`, 2D and 3D, InstanceNorm, LeakyReLU/ReLU, no dropout. Legacy and `architecture`-style plans, including `inherits_from`.
- One fold per bundle. Not supported (raises `NotImplementedError`): cascade configurations, region-based labels, ignore label.
- Checkpoints are loaded with `weights_only=True` plus a numpy-scalar allowlist; anything else in the pickle is refused rather than executed.

### Preprocessing scope

- Byte-equal to nnU-Net's `DefaultPreprocessor.run_case_npy` for the same array and spacing. Runs on CPU in numpy/scipy (float64 resampling, as nnU-Net does).
- The input array must use the axis order and spacing order nnU-Net's image reader produced for the model. The reader's name is recorded as `config.preprocessing.image_reader` (e.g. the nibabel readers return `(z, y, x)` with spacing reversed, `NibabelIOWithReorient` after reorienting to RAS). File loading and reorientation are the caller's job and are out of scope for this library.
- Normalization: `CTNormalization`, `ZScoreNormalization` (with and without mask), `RescaleTo01Normalization`, `RGBTo01Normalization`, `NoNormalization`.
- Resampling: `resample_data_or_seg_to_shape` only (nnU-Net's default); others raise `NotImplementedError`.
- 2D images need a singleton first spatial axis, as in nnU-Net.

## Testing

- `uv run pytest` — synthetic nnU-Net folders, no nnunetv2 needed.
- `tests/test_nnunet_equivalence.py` checks bitwise-equal logits against nnU-Net's own network (CPU FP32 and CUDA autocast), and `tests/test_nnunet_preprocessing_equivalence.py` checks byte-equal preprocessing against nnU-Net's `DefaultPreprocessor`. Both need nnunetv2 (the first also real model folders), so run them with the leadup environment:
  `PYTHONPATH=src packages/nnunet_model_conversion/.venv/bin/python -m pytest tests`
