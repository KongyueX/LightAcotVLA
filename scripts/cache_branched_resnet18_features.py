"""Cache frozen ImageNet ResNet-18 features for canonical branched images."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as functional
import torchvision.models as models
import tyro

from openpi.action_cot import branched_dataset


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    output_path: str
    batch_size: int = 256
    device: str = "cuda"


def _encode(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    original_shape = images.shape[:-3]
    flat = np.asarray(images).reshape((-1, *images.shape[-3:]))
    features: list[np.ndarray] = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            batch = torch.from_numpy(flat[start : start + batch_size]).to(
                device=device,
                dtype=torch.float32,
            )
            batch = batch.permute(0, 3, 1, 2) / 255.0
            batch = functional.interpolate(
                batch,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            batch = (batch - mean) / std
            features.append(model(batch).cpu().numpy().astype(np.float16))
    return np.concatenate(features).reshape((*original_shape, -1))


def main(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    arrays = branched_dataset.load_branched_arrays(
        args.dataset,
        fields=("root_id", "anchor_images", "current_images"),
    )
    device = torch.device(args.device)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    anchor_features = _encode(
        model,
        np.asarray(arrays["anchor_images"]),
        batch_size=args.batch_size,
        device=device,
    )
    current_features = _encode(
        model,
        np.asarray(arrays["current_images"]),
        batch_size=args.batch_size,
        device=device,
    )
    metadata = {
        "backbone": "torchvision_resnet18_imagenet1k_v1",
        "feature_dim": int(anchor_features.shape[-1]),
        "dtype": "float16",
        "root_count": int(anchor_features.shape[0]),
    }
    np.savez_compressed(
        output_path,
        root_id=np.asarray(arrays["root_id"]),
        anchor_features=anchor_features,
        current_features=current_features,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "output_path": str(output_path.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
