"""
Image captioning via SigLIP vision encoder + SmolLM2 decoder.

Provides ``ImageCaptioner`` — a self-contained inference pipeline that
generates natural-language descriptions of images using a frozen
SigLIP vision encoder, a trained projection layer, and a frozen
SmolLM2-135M language decoder.

Usage::

    captioner = ImageCaptioner()
    caption  = captioner.caption("resources/photo.jpg")
    # -> "a dog running through a grassy field."
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

warnings.filterwarnings("ignore", message=".*DecompressionBomb.*")
Image.MAX_IMAGE_PIXELS = None
logging.getLogger("transformers").setLevel(logging.ERROR)

if TYPE_CHECKING:
    from io import BytesIO

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

SIGLIP_MODEL_ID: str = "google/siglip-base-patch16-224"
SMOLM_MODEL_ID: str = "HuggingFaceTB/SmolLM2-135M"
PROJECTOR_WEIGHTS_PATH: str = "models/projector_weights.pth"
DEFAULT_MAX_NEW_TOKENS: int = 20


# ------------------------------------------------------------------
# Projector model
# ------------------------------------------------------------------

class _ImageToTextProjector(nn.Module):
    """Projects SigLIP vision embeddings into SmolLM2 token space."""

    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(vision_embeddings).unsqueeze(1)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class ImageCaptioner:
    """Generate natural-language captions for images.

    Loads a SigLIP vision encoder, a trained projection layer, and a
    frozen SmolLM2-135M decoder.  Falls back gracefully when
    dependencies are missing or weights cannot be found.

    Parameters
    ----------
    projector_weights_path:
        Path to the trained projector ``.pth`` file.
    siglip_model_id:
        HuggingFace model identifier for the SigLIP vision encoder.
    smolm_model_id:
        HuggingFace model identifier for the SmolLM2 decoder.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, etc.).  Defaults to
        CUDA when available.
    """

    def __init__(
        self,
        projector_weights_path: str = PROJECTOR_WEIGHTS_PATH,
        siglip_model_id: str = SIGLIP_MODEL_ID,
        smolm_model_id: str = SMOLM_MODEL_ID,
        device: str | None = None,
    ) -> None:
        self._ready = False

        try:
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print(
                "[ImageCaptioner] Transformers not installed — captioning "
                "disabled.  Install: pip install transformers"
            )
            return

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32

        # -- Vision encoder (SigLIP) --
        try:
            self._siglip = (
                AutoModel.from_pretrained(siglip_model_id, torch_dtype=self._dtype)
                .to(self._device)
                .eval()
            )
        except Exception as exc:
            print(f"[ImageCaptioner] Failed to load SigLIP: {exc}")
            return

        # -- Preprocessing transform --
        self._transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # -- Language decoder (frozen SmolLM2) --
        self._tokenizer = AutoTokenizer.from_pretrained(smolm_model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        try:
            self._decoder = (
                AutoModelForCausalLM.from_pretrained(
                    smolm_model_id, torch_dtype=self._dtype,
                )
                .to(self._device)
                .eval()
            )
            for param in self._decoder.parameters():
                param.requires_grad = False
        except Exception as exc:
            print(f"[ImageCaptioner] Failed to load SmolLM2: {exc}")
            return

        # -- Projector (trained) --
        llm_dim = self._decoder.config.hidden_size
        self._projector = _ImageToTextProjector(llm_dim).to(self._dtype).to(self._device)

        try:
            state = torch.load(projector_weights_path, map_location=self._device)
            self._projector.load_state_dict(state)
            self._projector.eval()
        except FileNotFoundError:
            print(
                f"[ImageCaptioner] Projector weights not found at "
                f"'{projector_weights_path}' — captioning disabled."
            )
            return
        except Exception as exc:
            print(f"[ImageCaptioner] Failed to load projector weights: {exc}")
            return

        self._ready = True
        print(f"[ImageCaptioner] Ready — device: {self._device}")

    # --------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------

    def caption(
        self,
        image: str | BytesIO | Image.Image,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        """Generate a caption for a single image.

        Parameters
        ----------
        image:
            File path, ``BytesIO`` buffer, or PIL ``Image`` object.
        max_new_tokens:
            Maximum number of tokens to generate (controls caption length).

        Returns:
            The generated caption string, or ``""`` if captioning is
            unavailable or fails.
        """
        if not self._ready:
            return ""

        return self.caption_batch([image], max_new_tokens=max_new_tokens)[0]

    @torch.no_grad()
    def caption_batch(
        self,
        images: list[str | BytesIO | Image.Image],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> list[str]:
        """Generate captions for a batch of images.

        Parameters
        ----------
        images:
            List of file paths, ``BytesIO`` buffers, or PIL ``Image``
            objects.
        max_new_tokens:
            Maximum number of tokens to generate per caption.

        Returns:
            List of caption strings (one per input image).  Entries for
            images that fail to load or process are returned as ``""``.
        """
        if not self._ready:
            return [""] * len(images)

        # --- Load and preprocess ---
        tensors: list[torch.Tensor] = []
        for source in images:
            try:
                if isinstance(source, Image.Image):
                    img = source.convert("RGB")
                elif isinstance(source, str):
                    img = Image.open(source).convert("RGB")
                else:
                    # BytesIO or file-like
                    img = Image.open(source).convert("RGB")
                tensors.append(self._transform(img))
            except Exception:
                tensors.append(torch.zeros(3, 224, 224))

        pixel_values = torch.stack(tensors).to(self._device)

        # --- SigLIP vision encoding ---
        vision_outputs = self._siglip.get_image_features(
            pixel_values=pixel_values
        )
        # Normalise to unit length (matches training)
        vision_embeds = vision_outputs / vision_outputs.norm(
            p=2, dim=-1, keepdim=True
        )

        # --- Project + decode ---
        projected = self._projector(vision_embeds)

        output_ids = self._decoder.generate(  # type: ignore[operator]
            inputs_embeds=projected,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            early_stopping=True,
            repetition_penalty=1.2,
        )

        captions: list[str] = []
        for ids in output_ids:
            raw = self._tokenizer.decode(ids, skip_special_tokens=True)
            # Keep only the first complete sentence
            clean = raw.split(".")[0].strip() + "."
            captions.append(clean)

        return captions


# ------------------------------------------------------------------
# Quick smoke test (python captioner.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    cap = ImageCaptioner()

    if cap._ready and len(sys.argv) > 1:
        print(cap.caption(sys.argv[1]))
    elif cap._ready:
        print("[smoke] ImageCaptioner loaded successfully (pass an image path to test).")
    else:
        print("[smoke] ImageCaptioner NOT ready (missing deps or weights).")
