# ragent-memory

Multimodal RAG (Retrieval-Augmented Generation) memory layer — ingest text documents and images into a persistent vector database, then search across both modalities with natural language queries powered by CLIP embeddings.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   DynamicAgentRAG                 │
│  add_to_memory(text)    add_batch_images(paths)   │
│  query_memory(q)        forget_memory(id)         │
└──────────┬──────────────────┬────────────────────┘
           │                  │
    ┌──────▼──────┐   ┌───────▼──────────────┐
    │   Chunker   │   │    EmbedManager       │
    │ chunk+meta  │   │  ┌──────────────────┐ │
    └──────┬──────┘   │  │ embed_text()     │ │
           │          │  │  PyTorch CPU FP32 │ │ ← live queries
           │          │  ├──────────────────┤ │
           │          │  │ embed_image()    │ │
           │          │  │  OpenVINO GPU     │ │ ← batch indexing
           │          │  │  FP16 (PyTorch    │ │
           │          │  │  CPU fallback)    │ │
           │          │  └──────────────────┘ │
           │          └──────────┬───────────┘
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

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Drop images in resources/ and text files in resources/

# 3. Run the interactive shell
python test_interface.py

# 4. Type a query
Write a search query: > a person standing outdoors at sunset
```

## Installation

```bash
# Core dependencies
pip install chromadb open_clip_torch torch Pillow

# OpenVINO GPU acceleration (optional — auto-falls back to CPU)
pip install openvino onnx onnxscript

# Intel Arc GPU acceleration (optional)
pip install intel-extension-for-pytorch
```

On first run the vision encoder is exported to ONNX and compiled for OpenVINO (one-time, cached at `models/vision_encoder.onnx`). Subsequent runs load it instantly.

## Usage

```python
from main import DynamicAgentRAG

rag = DynamicAgentRAG()

# --- Text documents ---
with open("resources/Engine.txt") as f:
    rag.add_to_memory(f.read())

# --- Images (batch — single forward pass per 32 images) ---
rag.add_batch_images_to_memory([
    "resources/photo1.jpg",
    "resources/photo2.jpg",
    "resources/diagram.png",
])

# --- Search across all modalities ---
result = rag.query_memory("exhaust manifold diagram")
print(rag.format_context(result["context"]))
# --- Result 1 (similarity: 0.760, source: abc123) ---
# The internal combustion engine is an engine in which...
#
# --- Result 2 (similarity: 0.642, source: def456) ---
# [IMAGE FILE] resources/engine_schematic.png

# --- Manage memory ---
rag.forget_memory("source_id")
print(rag.get_memory_stats())  # {"total_memories": 3, "total_chunks": 42}

# --- Check GPU status ---
rag.embedder.gpu_status()
```

## Project Structure

```
ragent-memory/
├── main.py                      # DynamicAgentRAG — top-level RAG orchestrator
├── embedder.py                  # EmbedManager — dual-backend CLIP embeddings
├── vision_encoder_openvino.py   # OpenVINO GPU (FP16) encoder + ONNX export
├── chunker.py                   # Chunker — text → overlapping chunks + metadata
├── dbmanager.py                 # DB — ChromaDB persistent client wrapper
├── test_interface.py            # Interactive CLI for testing
├── requirements.txt             # Direct Python dependencies
├── .gitignore
├── models/                      # Cached ONNX model (gitignored)
├── resources/                   # Your images and text files (gitignored)
└── vector_db/                   # ChromaDB persistent storage (gitignored)
```

## Dependencies

| Package | Purpose |
|---|---|
| `open_clip_torch` | ViT-SO400M-14-SigLIP-384 model |
| `chromadb` | Persistent vector database |
| `torch` / `torchvision` | PyTorch runtime + image transforms |
| `Pillow` | Image decoding |
| `openvino` | GPU-accelerated inference (optional) |
| `onnx` / `onnxscript` | Model export to OpenVINO (optional) |
| `intel-extension-for-pytorch` | Intel Arc iGPU for PyTorch (optional) |

When OpenVINO is unavailable, `embed_image` falls back to PyTorch CPU automatically.
