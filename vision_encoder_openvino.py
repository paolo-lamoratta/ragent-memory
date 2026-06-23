"""
OpenVINO-accelerated vision encoder for batch image embeddings.

Provides:
  - ONNX export of the vision encoder (one-time, cached to disk)
  - OpenVINO inference with FP16 precision on GPU
  - Graceful fallback when OpenVINO or GPU is unavailable

Usage
-----
    from vision_encoder_openvino import create_openvino_encoder

    ov_encoder = create_openvino_encoder(embed_manager, "models/vision.onnx")
    if ov_encoder:
        embeddings = ov_encoder.encode(image_tensor)  # numpy array
    else:
        # fall back to PyTorch
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------

def export_vision_encoder_to_onnx(
    embed_manager: Any,
    cache_path: str = "models/vitb_image_encoder.onnx",
    image_size: int = 224,
) -> str:
    """
    Export the vision encoder to ONNX for use with OpenVINO.

    Traces ``model.encode_image`` with a dummy input tensor of shape
    ``(1, 3, image_size, image_size)`` and writes the ONNX graph to
    *cache_path*.  The export is skipped if the file already exists.

    Args:
        embed_manager:  An ``EmbedManager`` instance whose ``.model``
                        holds the loaded model.
        cache_path:     Destination path for the ``.onnx`` file.
        image_size:     Expected input size in pixels (model-dependent).

    Returns:
        The absolute path to the ONNX file.
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    if os.path.exists(cache_path):
        return os.path.abspath(cache_path)

    model = embed_manager.model
    device = embed_manager.device

    # Dummy preprocessed image batch
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    # Trace encode_image — includes the forward pass + L2 normalisation
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input, # type: ignore[arg-type]
            cache_path,
            input_names=["pixel_values"],
            output_names=["image_embeddings"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "image_embeddings": {0: "batch"},
            },
            opset_version=18,
            verbose=False,
        )

    return os.path.abspath(cache_path)


# ------------------------------------------------------------------
# OpenVINO inference wrapper
# ------------------------------------------------------------------

class OpenVINOVisionEncoder:
    """
    Runs the vision encoder through OpenVINO with FP16 precision.

    Parameters
    ----------
    onnx_path : str
        Path to the exported ONNX model file.
    device : str
        OpenVINO device string (``"GPU"``, ``"CPU"``, etc.).
    """

    def __init__(self, onnx_path: str, device: str = "GPU") -> None:
        import openvino as ov

        self._device = device

        core = ov.Core()
        # Read the ONNX model directly — no separate IR conversion needed
        ov_model = core.read_model(onnx_path)
        # Set FP16 inference precision
        ov_model.set_rt_info("FP16", ["model_info", "inference_precision"])

        self._compiled = core.compile_model(ov_model, device, config={
            "PERFORMANCE_HINT": "THROUGHPUT",  # optimise for batched input
        })
        self._infer_request = self._compiled.create_infer_request()

    def encode(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Encode a batch of preprocessed images.

        Args:
            image_tensor:  ``(B, C, H, W)`` tensor on any device.
                           Will be moved to CPU numpy if needed.

        Returns:
            ``(B, embedding_dim)`` float32 numpy array (L2-normalised).
        """
        # OpenVINO expects a numpy array
        if image_tensor.device.type != "cpu":
            image_tensor = image_tensor.cpu()
        inputs = image_tensor.numpy()

        result = self._infer_request.infer({"pixel_values": inputs})
        return result["image_embeddings"]



# ------------------------------------------------------------------
# Factory — with graceful fallback
# ------------------------------------------------------------------

def create_openvino_encoder(
    embed_manager: Any,
    onnx_cache_path: str = "models/vision_encoder.onnx",
    preferred_device: str = "GPU",
) -> OpenVINOVisionEncoder | None:
    """
    Create an OpenVINO-accelerated vision encoder, falling back gracefully.

    Returns an ``OpenVINOVisionEncoder`` on success, or ``None`` if:
    - ``openvino`` is not installed
    - the ONNX file can't be loaded
    - the requested device is unavailable (tries ``"CPU"`` as fallback)

    Args:
        embed_manager:     An ``EmbedManager`` instance.
        onnx_cache_path:  Where the ONNX file lives / will be exported.
        preferred_device: OpenVINO device to target (``"GPU"`` or ``"CPU"``).

    Returns:
        ``OpenVINOVisionEncoder`` or ``None``.
    """
    # 1. Ensure ONNX model exists
    try:
        export_vision_encoder_to_onnx(embed_manager, onnx_cache_path)
    except Exception:
        return None

    # 2. Try loading with OpenVINO
    try:
        import openvino as ov
        core = ov.Core()
    except ImportError:
        return None
    except Exception:
        return None

    # 3. Pick device — prefer GPU, fall back to CPU
    device = preferred_device
    if device not in core.available_devices:
        if "CPU" in core.available_devices:
            device = "CPU"
        else:
            return None

    # 4. Compile
    try:
        encoder = OpenVINOVisionEncoder(onnx_cache_path, device=device)
        return encoder
    except Exception:
        return None
