from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .config import BUNDLE_FILE, WEIGHTS_FILE, BundleConfig
from .network import UNet, build_network


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _format_json(payload: dict[str, Any]) -> str:
    # Indented JSON with lists of scalars kept on one line, e.g. [3, 3, 3].
    text = json.dumps(payload, indent=2)
    text = re.sub(
        r"\[\s*([^\[\]{}]*?)\s*\]",
        lambda match: "[" + ", ".join(v.strip() for v in match.group(1).split(",") if v.strip()) + "]",
        text,
    )
    if json.loads(text) != json.loads(json.dumps(payload)):
        raise RuntimeError("Compact JSON formatting changed the bundle contents")
    return text + "\n"


def state_dict_mismatches(
    reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> list[str]:
    # Bitwise comparison: -0.0 vs 0.0 or differing NaN payloads count as differences.
    problems = [f"missing: {key}" for key in sorted(set(reference) - set(candidate))]
    problems += [f"unexpected: {key}" for key in sorted(set(candidate) - set(reference))]
    for key in sorted(set(reference) & set(candidate)):
        expected = reference[key].detach().cpu()
        actual = candidate[key].detach().cpu()
        if expected.dtype != actual.dtype:
            problems.append(f"dtype differs for {key}: {expected.dtype} vs {actual.dtype}")
        elif expected.shape != actual.shape:
            problems.append(f"shape differs for {key}: {tuple(expected.shape)} vs {tuple(actual.shape)}")
        elif not torch.equal(
            expected.contiguous().reshape(-1).view(torch.uint8),
            actual.contiguous().reshape(-1).view(torch.uint8),
        ):
            problems.append(f"values differ for {key}")
    return problems


@dataclass(eq=False)
class Bundle:
    config: BundleConfig
    network: UNet

    def save(self, output_dir: str | Path, *, overwrite: bool = False) -> Path:
        output = Path(output_dir)
        bundle_path = output / BUNDLE_FILE
        weights_path = output / WEIGHTS_FILE
        if not overwrite and (bundle_path.exists() or weights_path.exists()):
            raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")
        output.mkdir(parents=True, exist_ok=True)

        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.network.state_dict().items()
        }
        # A bundle without bundle.json is not loadable, so drop a stale one first
        # and write the new one last.
        bundle_path.unlink(missing_ok=True)
        temporary = weights_path.with_name(f".{WEIGHTS_FILE}.tmp")
        try:
            save_file(state, str(temporary))
            mismatches = state_dict_mismatches(state, load_file(str(temporary), device="cpu"))
            if mismatches:
                raise RuntimeError(
                    "safetensors weights are not bitwise identical to the network:\n  "
                    + "\n  ".join(mismatches[:20])
                )
            os.replace(temporary, weights_path)
        finally:
            temporary.unlink(missing_ok=True)

        payload = {
            **self.config.to_dict(),
            "weights": {"file": WEIGHTS_FILE, "sha256": _sha256(weights_path)},
        }
        temporary = bundle_path.with_name(f".{BUNDLE_FILE}.tmp")
        temporary.write_text(_format_json(payload))
        os.replace(temporary, bundle_path)
        return output


def load_bundle(bundle_dir: str | Path, *, device: str | torch.device = "cpu") -> Bundle:
    bundle = Path(bundle_dir)
    raw: dict[str, Any] = json.loads((bundle / BUNDLE_FILE).read_text())
    config = BundleConfig.from_dict(raw)
    weights_path = bundle / str(raw["weights"]["file"])
    if not weights_path.is_file():
        raise FileNotFoundError(f"Bundle weights not found: {weights_path}")
    if _sha256(weights_path) != raw["weights"]["sha256"]:
        raise ValueError(f"Weights SHA-256 does not match {BUNDLE_FILE}: {weights_path}")
    network = build_network(config.network)
    network.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
    network.eval().to(torch.device(device))
    return Bundle(config=config, network=network)
