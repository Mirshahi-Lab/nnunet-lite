"""End-to-end prediction on synthetic bundles: inputs, output grid, QC and multi-bundle runs."""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from test_convert import make_model_dir

import nnunet_lite as nl
from nnunet_lite import predict as predict_module
from nnunet_lite.sliding_window import predict_logits

SPACING = (3.0, 1.2, 1.5)


@pytest.fixture
def bundle(tmp_path: Path) -> nl.Bundle:
    return nl.convert(make_model_dir(tmp_path, "3d_fullres")[0])


def volume(shape: tuple[int, int, int] = (14, 30, 26), border: int = 3) -> np.ndarray:
    data = np.zeros((1, *shape), dtype=np.float32)
    inner = tuple(slice(border, size - border) for size in shape)
    values = np.random.default_rng(0).uniform(-150, 250, size=data[(0, *inner)].shape)
    data[(0, *inner)] = values
    return data


def nifti(data: np.ndarray, spacing: tuple[float, ...]) -> nib.Nifti1Image:
    # (Z, Y, X) array -> RAS file, which the reorienting reader returns unchanged.
    return nib.Nifti1Image(data[0].transpose(2, 1, 0), np.diag([*spacing[::-1], 1.0]))


def with_preprocessing(bundle: nl.Bundle, **changes: object) -> nl.Bundle:
    preprocessing = replace(bundle.config.preprocessing, **changes)  # type: ignore[arg-type]
    return nl.Bundle(replace(bundle.config, preprocessing=preprocessing), bundle.network)


def test_segmentation_is_on_the_input_grid(bundle: nl.Bundle) -> None:
    data = volume()
    predictor = nl.Predictor(bundle, device="cpu")
    result = predictor.predict(data, SPACING)
    assert result.segmentation.shape == data.shape[1:]
    assert result.segmentation.dtype == torch.uint8
    assert result.segmentation.device.type == "cpu"
    assert result.labels == {0: "background", 1: "organ", 2: "lesion"}
    assert result.qc is None
    assert set(result.segmentation.unique().tolist()) <= {0, 1, 2}
    # Everything outside the nonzero crop is background.
    outside = torch.ones(data.shape[1:], dtype=torch.bool)
    outside[3:-3, 3:-3, 3:-3] = False
    assert not result.segmentation[outside].any()
    assert torch.equal(
        predictor.predict(torch.from_numpy(data), SPACING).segmentation, result.segmentation
    )


def test_matches_the_individual_steps_without_resampling(bundle: nl.Bundle) -> None:
    data = volume()
    spacing = bundle.config.preprocessing.target_spacing
    tensor, properties = nl.preprocess(data, spacing, bundle.config.preprocessing, device="cpu")
    expected = torch.zeros(data.shape[1:], dtype=torch.uint8)
    box = tuple(slice(start, end) for start, end in properties.crop_bbox)
    expected[box] = predict_logits(bundle.network, tensor, bundle.config).argmax(0).to(torch.uint8)
    result = nl.Predictor(bundle, device="cpu").predict(data, spacing)
    assert torch.equal(result.segmentation, expected)


def test_transposed_models_return_the_input_axis_order(bundle: nl.Bundle) -> None:
    data = volume()
    forward, backward = (2, 0, 1), (1, 2, 0)
    transposed = with_preprocessing(bundle, transpose_forward=forward, transpose_backward=backward)
    result = nl.Predictor(transposed, device="cpu").predict(data, SPACING)
    assert result.segmentation.shape == data.shape[1:]
    # Same as feeding the plain model the transposed array and transposing its answer back.
    plain = nl.Predictor(bundle, device="cpu").predict(
        data.transpose(0, *[axis + 1 for axis in forward]), [SPACING[axis] for axis in forward]
    )
    assert torch.equal(result.segmentation, plain.segmentation.permute(*backward))


def test_file_input_matches_array_input(bundle: nl.Bundle, tmp_path: Path) -> None:
    data = volume()
    image = nifti(data, SPACING)
    path = tmp_path / "ct.nii.gz"
    nib.save(image, str(path))
    predictor = nl.Predictor(bundle, device="cpu")
    reference = predictor.predict(data, SPACING).segmentation
    for source in (path, str(path), path.read_bytes(), io.BytesIO(path.read_bytes()), image, [path]):
        result = predictor.predict(source)
        assert torch.equal(result.segmentation, reference)

    reloaded = nib.Nifti1Image.from_bytes(result.to_bytes(compress=False))
    assert reloaded.shape == image.shape
    assert np.allclose(reloaded.affine, image.affine)
    assert np.array_equal(np.asarray(reloaded.dataobj), reference.numpy().transpose(2, 1, 0))
    saved = result.save(tmp_path / "seg.nii.gz")
    assert np.array_equal(np.asarray(nib.load(str(saved)).dataobj), np.asarray(reloaded.dataobj))


def test_file_output_keeps_the_orientation_of_the_input(bundle: nl.Bundle) -> None:
    data = volume()
    ras = nifti(data, SPACING)
    lps = ras.as_reoriented(nib.orientations.axcodes2ornt("LPS"))
    predictor = nl.Predictor(bundle, device="cpu")
    from_ras = nib.Nifti1Image.from_bytes(predictor.predict(ras).to_bytes(compress=False))
    from_lps = nib.Nifti1Image.from_bytes(predictor.predict(lps).to_bytes(compress=False))
    assert np.allclose(from_lps.affine, lps.affine)
    assert nib.aff2axcodes(from_lps.affine) == ("L", "P", "S")
    assert np.array_equal(np.asarray(from_lps.dataobj), np.asarray(from_ras.dataobj)[::-1, ::-1])


def test_input_validation(bundle: nl.Bundle) -> None:
    data = volume()
    predictor = nl.Predictor(bundle, device="cpu")
    with pytest.raises(ValueError, match="spacing is required"):
        predictor.predict(data)
    with pytest.raises(ValueError, match="cannot be overridden"):
        predictor.predict(nifti(data, SPACING), SPACING)
    with pytest.raises(ValueError, match="expects 1 channel"):
        predictor.predict([nifti(data, SPACING)] * 2)
    with pytest.raises(ValueError, match="expects 1 channel"):
        predictor.predict(np.concatenate([data, data]), SPACING)
    with pytest.raises(ValueError, match="Expected a NIfTI path"):
        predictor.predict([data[0]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="there is no image geometry"):
        predictor.predict(data, SPACING).save("seg.nii.gz")
    with pytest.raises(ValueError, match="batch_size"):
        nl.Predictor(bundle, batch_size=0)
    with pytest.raises(NotImplementedError, match="SimpleITKIO"):
        nl.Predictor(with_preprocessing(bundle, image_reader="SimpleITKIO"), device="cpu").predict(
            nifti(data, SPACING)
        )
    # The same model still takes arrays, whatever reader it was trained with.
    nl.Predictor(with_preprocessing(bundle, image_reader="SimpleITKIO"), device="cpu").predict(
        data, SPACING
    )


def test_qc_is_opt_in_and_per_label(bundle: nl.Bundle) -> None:
    data = volume()
    predictor = nl.Predictor(bundle, device="cpu")
    result = predictor.predict(data, SPACING, qc=True)
    assert result.qc is not None
    assert list(result.qc) == ["organ", "lesion"]
    for index, name in ((1, "organ"), (2, "lesion")):
        record = result.qc[name]
        assert isinstance(record, nl.LabelQC)
        assert record.voxel_count == int((result.segmentation == index).sum())
        assert 0.0 <= record.estimated_dice <= 1.0
        assert 0.0 <= record.mean_entropy <= 1.0
        assert 0.0 <= record.mean_margin <= 1.0
        assert 0.0 <= record.low_margin_fraction <= 1.0
    assert torch.equal(result.segmentation, predictor.predict(data, SPACING).segmentation)

    selected = predictor.predict(data, SPACING, qc=["lesion"])
    assert selected.qc == {"lesion": result.qc["lesion"]}
    assert predictor.predict(data, SPACING, qc=[2]).qc == selected.qc
    assert predictor.predict(data, SPACING, qc=[]).qc == {}
    with pytest.raises(ValueError, match="Unknown label"):
        predictor.predict(data, SPACING, qc=["spleen"])


def test_always_preprocesses_with_the_bundle_configuration(
    bundle: nl.Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    preprocess = predict_module.preprocess

    def recording(*args: object, **kwargs: object) -> object:
        calls.append(args[2])
        return preprocess(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(predict_module, "preprocess", recording)
    nl.Predictor(bundle, device="cpu").predict(volume(), SPACING)
    assert calls == [bundle.config.preprocessing]


def test_2d_bundle(tmp_path: Path) -> None:
    bundle = nl.convert(make_model_dir(tmp_path, "2d")[0])
    data = volume((1, 30, 26), border=0)
    result = nl.Predictor(bundle, device="cpu").predict(data, (999.0, 1.4, 1.0), qc=True)
    assert result.segmentation.shape == (1, 30, 26)
    assert result.qc is not None and set(result.qc) == {"organ", "lesion"}


def test_multi_predictor_matches_single_predictors(
    bundle: nl.Bundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = volume()
    channel = replace(bundle.config.preprocessing.channels[0], lower=-20.0, upper=80.0)
    bundles = {
        "a": bundle,
        "clipped": with_preprocessing(bundle, channels=(channel,)),
        "b": nl.Bundle(replace(bundle.config, dataset="Dataset002_Other"), bundle.network),
    }
    path = tmp_path / "ct.nii.gz"
    nib.save(nifti(data, SPACING), str(path))

    calls = []
    preprocess = predict_module.preprocess

    def recording(*args: object, **kwargs: object) -> object:
        calls.append(args[2])
        return preprocess(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(predict_module, "preprocess", recording)
    multi = nl.MultiPredictor(bundles, device="cpu")
    results = multi.predict(io.BytesIO(path.read_bytes()), qc={"a": True, "clipped": ["organ"]})
    # "a" and "b" share their preprocessing; "clipped" needs its own.
    assert len(calls) == 2
    assert list(results) == ["a", "clipped", "b"]
    assert results["a"].qc is not None and list(results["a"].qc) == ["organ", "lesion"]
    assert results["clipped"].qc is not None and list(results["clipped"].qc) == ["organ"]
    assert results["b"].qc is None

    for name, single_bundle in bundles.items():
        single = nl.Predictor(single_bundle, device="cpu").predict(path, qc=True)
        assert torch.equal(results[name].segmentation, single.segmentation)
        assert results[name].to_bytes() == single.to_bytes()
        for label, record in (results[name].qc or {}).items():
            assert single.qc is not None and record == single.qc[label]
    assert not torch.equal(results["a"].segmentation, results["clipped"].segmentation)

    from_array = multi.predict(data, SPACING, qc=True)
    assert all(torch.equal(from_array[n].segmentation, results[n].segmentation) for n in bundles)
    assert all(r.qc is not None for r in from_array.values())
    with pytest.raises(ValueError, match="unknown bundles"):
        multi.predict(data, SPACING, qc={"missing": True})
    with pytest.raises(ValueError, match="at least one bundle"):
        nl.MultiPredictor({})


def test_loads_bundles_from_directories(tmp_path: Path) -> None:
    nl.convert(make_model_dir(tmp_path, "3d_fullres")[0], output_dir=tmp_path / "bundle")
    data = volume()
    single = nl.Predictor(tmp_path / "bundle", device="cpu").predict(data, SPACING)
    multi = nl.MultiPredictor({"x": tmp_path / "bundle"}, device="cpu").predict(data, SPACING)
    assert torch.equal(single.segmentation, multi["x"].segmentation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_prediction(bundle: nl.Bundle) -> None:
    data = volume()
    cpu = nl.Predictor(bundle, device="cpu").predict(data, SPACING, qc=True)
    predictor = nl.Predictor(bundle)  # default device
    assert predictor.device.type == "cuda"
    gpu = predictor.predict(data, SPACING, qc=True)
    assert gpu.segmentation.device.type == "cpu"
    assert gpu.segmentation.shape == cpu.segmentation.shape
    # Random weights give near-tied logits, so only rough agreement can be expected.
    assert (gpu.segmentation == cpu.segmentation).float().mean() > 0.9
    assert gpu.qc is not None and cpu.qc is not None
    assert gpu.qc["organ"].estimated_dice == pytest.approx(cpu.qc["organ"].estimated_dice, abs=0.05)
