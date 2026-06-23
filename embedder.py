"""
EmbedManager: multimodal embedding via OpenCLIP.

Provides two separate embedding pipelines:
  - embed_text()   → for user search queries (text encoder)
  - embed_image()  → for ingesting image files (vision encoder)
                      Uses OpenVINO GPU (FP16) when available, falls back to
                      PyTorch CPU otherwise.

Both pipelines apply L2 normalisation, which is mandatory for
cosine-similarity comparisons in the vector database.
"""

import io
import os
import time
import logging
import contextlib
import concurrent.futures
import warnings

# Suppress noisy library output before any model imports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings("ignore", message=".*DecompressionBomb.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch

# Attempt to enable Intel Arc iGPU acceleration via IPEX.
# Gracefully fall back to CPU when IPEX is absent or its version
# doesn't match the installed PyTorch.
with contextlib.redirect_stderr(io.StringIO()):
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        _IPEX_AVAILABLE = True
    except Exception:
        _IPEX_AVAILABLE = False

with contextlib.redirect_stderr(io.StringIO()):
    import open_clip
    from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # disable decompression-bomb check


class EmbedManager():
    """
    Multimodal embedding manager powered by OpenCLIP.

    Auto-detects Intel Arc iGPU (xpu backend) when available and falls back
    to CPU otherwise.  Exposes separate ``embed_text`` and ``embed_image``
    entry points so callers can route text queries and image ingestion
    through the correct encoder.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16-SigLIP",#"ViT-SO400M-14-SigLIP-384",
        pretrained: str = "webli",
        onnx_cache_path: str = "models/vitb_image_encoder.onnx",
    ) -> None:
        """
        Load the OpenCLIP model, preprocessing transforms, and tokenizer.
        Also attempts to create an OpenVINO-accelerated vision encoder.

        Args:
            model_name:      OpenCLIP model identifier.
            pretrained:      Pretrained weights tag (e.g. ``"webli"``).
            onnx_cache_path: Where to cache the exported ONNX model.
        """
        # --- Device selection: prefer Intel Arc iGPU, fallback to CPU ---
        self.device = (
            "xpu"
            if hasattr(torch, "xpu") and torch.xpu.is_available()
            else "cpu"
        )
        # --- Model, image-preprocessing pipeline, and tokenizer ---
        with contextlib.redirect_stderr(io.StringIO()):
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device,
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

        # --- OpenVINO vision encoder (FP16 GPU for batch image indexing) ---
        self._ov_encoder = None
        try:
            from vision_encoder_openvino import create_openvino_encoder
            self._ov_encoder = create_openvino_encoder(self, onnx_cache_path)
        except Exception:
            pass

        if self._ov_encoder:
            print("[EmbedManager] Vision encoder: GPU (OpenVINO, FP16)")
        else:
            print(f"[EmbedManager] Vision encoder: {self.device} (PyTorch, FP32)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str | list[str]) -> list[list[float]]:
        """
        Encode one or more text strings through the text encoder.

        Args:
            text:  A single query string or a batch of strings.
                   NOTE — the tokenizer may truncate long inputs.

        Returns:
            A list of L2-normalised embedding vectors (one per input string),
            each represented as ``list[float]``.
        """
        if isinstance(text, str):
            text = [text]

        t0 = time.perf_counter()

        # Tokenize and move to the active device
        text_tokens = self.tokenizer(text).to(self.device)

        with torch.inference_mode():
            text_features = self.model.encode_text(text_tokens)
            # L2 normalisation — required for cosine-similarity searches
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.tolist()

    def embed_image(self, image_paths: str | list[str]) -> list[list[float]]:
        """
        Encode one or more image files through the vision encoder.

        Routes through OpenVINO GPU (FP16) when available; falls back to
        PyTorch CPU (FP32) otherwise.

        Args:
            image_paths:  Path(s) to image file(s) on disk.

        Returns:
            A list of L2-normalised embedding vectors (one per image),
            each represented as ``list[float]``.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        t0 = time.perf_counter()

        # Decode and preprocess images in parallel via a thread pool.
        # PIL/JPEG decoding is I/O-bound, so 16 workers keep the pipeline
        # saturated — while one batch goes through the encoder, the next
        # batch of raw images is already being decoded.
        def _load_and_preprocess(path: str):
            return self.preprocess(Image.open(path).convert("RGB"))

        n_workers = min(len(image_paths), 24)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            images = list(pool.map(_load_and_preprocess, image_paths))

        image_input = torch.stack(images)

        # --- OpenVINO path (GPU, FP16) ---
        if self._ov_encoder is not None:
            # image_input stays on CPU — OpenVINO reads numpy directly
            embeddings = self._ov_encoder.encode(image_input)
            return embeddings.tolist()

        # --- PyTorch fallback (CPU, FP32) ---
        image_input = image_input.to(self.device)
        with torch.inference_mode():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.tolist()

    def gpu_status(self) -> None:
        """Print current GPU state — call any time to verify GPU is live."""
        from vision_encoder_openvino import print_gpu_status
        print_gpu_status(self._ov_encoder)


# ------------------------------------------------------------------
# Quick smoke test (run with:  python embedder.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    client = EmbedManager()

    # --- Text embedding ---
    sample_text = "Hi everybody, this is the embedding vector of this string"
    text_vecs = client.embed_text(sample_text)
    print(
        f"\nText embedding shape: 1 × {len(text_vecs[0])}\n"
        f"First 8 dims: {text_vecs[0][:8]}"
    )

    # --- Image embedding (uncomment and point to a real image to test) ---
    # import sys
    # if len(sys.argv) > 1:
    #     img_vecs = client.embed_image(sys.argv[1])
    #     print(f"\nImage embedding shape: 1 × {len(img_vecs[0])}")
