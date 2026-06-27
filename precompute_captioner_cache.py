#!/usr/bin/env python3
"""
Precompute SigLIP embeddings for captioner training (CPU-only).

Runs the SigLIP vision encoder on all COCO 2017 training images once and
saves the resulting embeddings alongside tokenized captions to
``siglip_cache.pt``.  This cache is then consumed by the projector training
step (see ``train_captioner.ipynb``).

Usage::

    python3 precompute_captioner_cache.py

Dataset: COCO 2017 (~18 GB, auto-downloads if missing).
"""

import os
import sys
import multiprocessing
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from datasets import load_dataset
from tqdm.auto import tqdm


# 1. DEVICE / DATA HELPERS

def get_device() -> torch.device:
    """Return the CPU device."""
    device = torch.device("cpu")
    print(f"[Device] Using CPU ({multiprocessing.cpu_count()} cores)")
    return device


COCO_DIR = "train2017"
COCO_ZIP = "train2017.zip"
COCO_URL = "http://images.cocodataset.org/zips/train2017.zip"
CACHE_FILE = "siglip_cache.pt"


def ensure_coco_downloaded():
    """Download & extract COCO 2017 training images if the folder is missing."""
    if os.path.isdir(COCO_DIR) and len(os.listdir(COCO_DIR)) > 0:
        print(f"[Data] COCO images found in '{COCO_DIR}/' — skipping download.")
        return

    print("=" * 60)
    print("  COCO 2017 training images not found.")
    print(f"  Will download ~18 GB from {COCO_URL}")
    print("  This may take 5-15 minutes depending on your connection.")
    print("=" * 60)

    import subprocess
    ret = subprocess.run(
        ["wget", "-c", COCO_URL],
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    if ret.returncode != 0:
        print("[Data] ERROR: Download failed. Check your connection and try again.")
        sys.exit(1)

    print("[Data] Extracting zip file (this will take a few minutes)...")
    import zipfile
    with zipfile.ZipFile(COCO_ZIP, "r") as zf:
        zf.extractall()
    print("[Data] Extraction complete!")
    subprocess.run(["rm", "train2017.zip"])
    print(f"[Data] Ready. Images in '{COCO_DIR}/'.")
    print("[Data] Deleted zip archive after extraction.")


def resolve_image_path(file_name: str) -> str:
    """COCO file_name may or may not include the train2017/ prefix."""
    if os.path.exists(file_name):
        return file_name
    candidate = os.path.join(COCO_DIR, file_name)
    if os.path.exists(candidate):
        return candidate
    return candidate


# 2. PRECOMPUTE

def _extract_vision_vectors(vision_outputs):
    """Handle different SigLIP output formats across transformers versions."""
    if hasattr(vision_outputs, "image_embeds"):
        embeds = vision_outputs.image_embeds
    elif hasattr(vision_outputs, "pooler_output"):
        embeds = vision_outputs.pooler_output
    else:
        embeds = vision_outputs
    return embeds / embeds.norm(p=2, dim=-1, keepdim=True)


def precompute_embeddings():
    """Run SigLIP on all COCO images once, save embeddings + tokenized captions."""
    device = get_device()
    dtype = torch.float32

    ensure_coco_downloaded()

    # -- SigLIP --
    print("[Precompute] Loading SigLIP processor & vision encoder...")
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    siglip_model = AutoModel.from_pretrained(
        "google/siglip-base-patch16-224", dtype=dtype,
    ).to(device).eval()
    print("[Precompute] SigLIP encoder loaded.")

    # -- Tokenizer for captions --
    print("[Precompute] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Dataset --
    print("[Precompute] Loading COCO captions...")
    dataset = load_dataset("phiyodr/coco2017", split="train")

    # No DataLoader workers — eliminate IPC overhead.  Single-process,
    # bigger batches (256) so the model spends more time computing and
    # less time waiting for the next batch.
    batch_size = 512
    torch.set_num_threads(multiprocessing.cpu_count())  # use all 22 cores
    all_embeddings = []
    all_token_ids = []
    image_buffer = []
    text_buffer = []

    print(f"[Precompute] Processing {len(dataset):,} images "
          f"(batch_size={batch_size}, single-process)...")

    with torch.no_grad():
        for item in tqdm(dataset, desc="Precompute", total=len(dataset)):
            img_path = resolve_image_path(item["file_name"])
            with Image.open(img_path) as img:
                image_buffer.append(img.convert("RGB"))
            text_buffer.append(item["captions"][0])

            if len(image_buffer) >= batch_size:
                # Preprocess & run SigLIP in one batch
                inputs = siglip_processor(images=image_buffer, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(dtype).to(device)

                vision_outputs = siglip_model.get_image_features(
                    pixel_values=pixel_values
                )
                vision_vectors = _extract_vision_vectors(vision_outputs)
                all_embeddings.append(vision_vectors.cpu())

                # Tokenize captions for this batch
                tok = tokenizer(
                    text_buffer,
                    padding="max_length", max_length=40,
                    truncation=True, return_tensors="pt",
                ).input_ids
                all_token_ids.append(tok)

                image_buffer.clear()
                text_buffer.clear()

        # Remainder batch
        if image_buffer:
            inputs = siglip_processor(images=image_buffer, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype).to(device)
            vision_outputs = siglip_model.get_image_features(
                pixel_values=pixel_values
            )
            vision_vectors = _extract_vision_vectors(vision_outputs)
            all_embeddings.append(vision_vectors.cpu())

            tok = tokenizer(
                text_buffer,
                padding="max_length", max_length=40,
                truncation=True, return_tensors="pt",
            ).input_ids
            all_token_ids.append(tok)

    embeddings = torch.cat(all_embeddings, dim=0)   # (N, 768)
    token_ids  = torch.cat(all_token_ids,  dim=0)   # (N, 40)

    size_mb = embeddings.element_size() * embeddings.numel() / (1024 ** 2)
    print(f"[Precompute] Saving {embeddings.shape[0]:,} embeddings "
          f"({size_mb:.0f} MB) to '{CACHE_FILE}'...")
    torch.save({"embeddings": embeddings, "token_ids": token_ids}, CACHE_FILE)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    precompute_embeddings()
