# nnunet-lite

Convert an nnU-Net model into a simple bundle (`bundle.json` + `weights.safetensors`) and run it without nnU-Net: tensor in, tensor out.

## Preprocessing: CPU vs GPU

`nl.preprocess` runs nnU-Net's preprocessing (crop to nonzero → normalize → resample) on a raw array. It has two paths:

```python
tensor, props = nl.preprocess(volume, spacing, config.preprocessing)  # GPU if available
tensor, props = nl.preprocess(volume, spacing, config.preprocessing, device="cpu")  # exact nnU-Net path
```

| | CPU (`device="cpu"`) | GPU (default when CUDA is available) |
|---|---|---|
| Match to nnU-Net | Byte-equal | Within float32 rounding |
| Largest difference from the CPU path | — | 5e-7 on normalized data (one float32 rounding step) |
| Voxels bit-identical to the CPU path | 100% | about 93% |
| Crop box and output shape | Same as nnU-Net | Identical to the CPU path |

### Why the GPU path can be trusted

- **Same math, not a reimplementation.** Spline resampling is linear and separable, so the GPU applies it as one float64 matrix product per axis. Those matrices are produced by the same scipy call the CPU path makes. The only difference is the order in which floating-point sums are added up.
- **Tested against the CPU path** on every combination of 7 configurations (CT, masked and plain z-score, rescale, 2D, transposed axes, linear resampling) and 6 geometries (isotropic, thick slices on each axis, odd spacings, input already at target spacing).
- **The CPU path is tested against nnU-Net itself**: byte-equal to `DefaultPreprocessor.run_case_npy` on 138 synthetic plan/geometry combinations, 12 runs using the real plans of four trained models, and three real CT scans.
- **Automatic fallback.** If the GPU path fails (out of memory, CUDA error, NaN/inf in the input), a `RuntimeWarning` is emitted and the exact CPU path runs instead.

### Speed

Measured on an RTX 2080 Ti with synthetic volumes at realistic sizes, resampling to 1.5 mm:

| Input | CPU | GPU | Speedup | Peak GPU memory |
|---|---|---|---|---|
| 300×512×512 at 0.8 mm | 8.76 s | 0.30 s | 29× | 1.3 GB |
| 80×512×512, 5 mm slices | 2.32 s | 0.11 s | 22× | 0.6 GB |
| 600×512×512 at 1×0.7 mm | 17.68 s | 0.45 s | 39× | 3.6 GB |
| 2-channel MRI 150×240×240, masked z-score | 2.82 s | 0.24 s | 12× | 0.3 GB |

### Which to use

Use the default for throughput. Use `device="cpu"` when you need output that is bit-for-bit what nnU-Net would produce, for example when validating a converted model against nnU-Net.

The GPU differences are at the level of float32 rounding in the network's *input*. Their effect on final segmentations has not been measured yet; that comparison belongs with the inference module.
