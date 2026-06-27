# ragent-memory

Multimodal RAG (Retrieval-Augmented Generation) memory layer — ingest text documents and images into a persistent vector database, then search across both modalities with natural language queries powered by multimodal embeddings.

Images are auto-captioned via a SigLIP→SmolLM2 pipeline, so text queries like *"a dog running in a park"* find photos even when the visual embedding alone wouldn't surface them.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   DynamicAgentRAG                     │
│  add_to_memory(text)    add_batch_images(paths)       │
│  query_memory(q)        forget_memory(id)             │
└──────────┬──────────────────┬────────────────────────┘
           │                  │
    ┌──────▼──────┐   ┌───────▼──────────────────────┐
    │   Chunker   │   │       EmbedManager            │
    │ chunk+meta  │   │  ┌───────────────────────────┐│
    └──────┬──────┘   │  │ embed_text()  — PyTorch   ││ ← live queries
           │          │  │ embed_image() — OpenVINO   ││ ← batch indexing
           │          │  │                GPU FP16    ││
           │          │  └───────────────────────────┘│
           │          │  ┌───────────────────────────┐│
           │          │  │ ImageCaptioner             ││
           │          │  │ SigLIP→SmolLM2→caption     ││ ← auto-captions
           │          │  └───────────────────────────┘│
           │          └──────────┬───────────────────┘
           │                     │
    ┌──────▼─────────────────────▼──────┐
    │            ChromaDB                │
    │     (persistent, cosine space)     │
    └────────────────────────────────────┘
```

| Pipeline Stage | Backend | Precision | Purpose |
|---|---|---|---|
| Image indexing (batch) | OpenVINO GPU | FP16 | Throughput — up to 35 img/s |
| Text queries (live) | PyTorch CPU | FP32 | Low overhead per query |
| Image captioning | SigLIP + SmolLM2 | FP32 | Auto-generated descriptions |

## Quick Start

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. (Optional) Install captioning support
pip install transformers datasets tqdm

# 3. Drop images and text files in resources/

# 4. Run the interactive shell
python scripts/test_interface.py

# 5. Type a query
> a person standing outdoors at sunset
```

## Installation

```bash
# Core
pip install chromadb open_clip_torch torch Pillow

# OpenVINO GPU acceleration (optional — auto-falls back to CPU)
pip install openvino onnx onnxscript

# Image captioning (optional — falls back gracefully)
pip install transformers datasets tqdm
```

The vision encoder is exported to ONNX and compiled for OpenVINO on first use (one-time, cached at `models/vitb_image_encoder.onnx`).

## Usage

```python
from ragent_memory import DynamicAgentRAG

rag = DynamicAgentRAG()

# --- Text documents ---
with open("resources/report.txt") as f:
    rag.add_to_memory(f.read())

# --- Images (auto-captioned) ---
rag.add_batch_images_to_memory([
    "resources/photo1.jpg",
    "resources/photo2.jpg",
    "resources/diagram.png",
])

# --- Search across all modalities ---
result = rag.query_memory("exhaust manifold diagram")
print(rag.format_context(result["context"]))
# --- Result 1 (similarity: 0.823, source: abc123) ---
# The internal combustion engine is an engine in which...
#
# --- Result 2 (similarity: 0.761, source: def456) ---
# [IMAGE] "a diagram of an exhaust manifold" → resources/engine_diagram.png

# --- Manage memory ---
rag.forget_memory("source_id")
print(rag.get_memory_stats())  # {"total_memories": 3, "total_chunks": 42}
```

### Captioning details

Set `enable_captioning=False` to skip caption generation:

```python
rag = DynamicAgentRAG(enable_captioning=False)
rag.add_image_to_memory("photo.jpg")  # no caption generated
```

The captioner uses a frozen SigLIP vision encoder + a trained projector + a frozen SmolLM2-135M decoder. Weights live in `models/projector_weights.pth`. When `transformers` is not installed or weights are missing, captioning is silently skipped.

## Project Structure

```
ragent-memory/
├── ragent_memory/                   # Python package
│   ├── __init__.py
│   ├── core.py                      # DynamicAgentRAG — top-level RAG orchestrator
│   ├── embedder.py                  # EmbedManager — dual-backend multimodal embeddings
│   ├── captioner.py                 # ImageCaptioner — SigLIP→SmolLM2 caption inference
│   ├── vision_encoder_openvino.py   # OpenVINO GPU (FP16) encoder + ONNX export
│   ├── chunker.py                   # Chunker — text → overlapping chunks + metadata
│   ├── dbmanager.py                 # DB — ChromaDB persistent client wrapper
│   └── loader.py                    # DocumentLoader — PDF / DOCX text extraction
├── scripts/                         # Runnable entry points
│   ├── test_interface.py            # Interactive CLI for testing
│   └── precompute_captioner_cache.py # Step 1 of captioner training
├── models/                          # Projector weights (tracked) + ONNX cache (gitignored)
├── resources/                       # Your images and text files (gitignored)
├── vector_db/                       # ChromaDB persistent storage (gitignored)
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

### Training scripts

The captioning projector was trained on COCO 2017. The two-step training pipeline is preserved for reproducibility:

```
scripts/precompute_captioner_cache.py   ← Step 1: Run SigLIP on COCO → siglip_cache.pt
train_captioner.ipynb                   ← Step 2: Train projector → models/projector_weights.pth
```

These are not needed for inference — just the weights file.

## Dependencies

| Package | Purpose |
|---|---|
| `open_clip_torch` | OpenCLIP model (configurable) |
| `chromadb` | Persistent vector database |
| `torch` / `torchvision` | PyTorch runtime + image transforms |
| `Pillow` | Image decoding |
| `PyMuPDF` / `python-docx` | Document parsing (PDF, DOCX) |
| `transformers` | SigLIP + SmolLM2 for captioning (optional) |
| `datasets` / `tqdm` | Training dataset and progress bars (optional) |
| `openvino` / `onnx` / `onnxscript` | GPU-accelerated inference (optional) |

When OpenVINO is unavailable, `embed_image` falls back to PyTorch CPU automatically.
When `transformers` or projector weights are unavailable, captioning is silently skipped.
