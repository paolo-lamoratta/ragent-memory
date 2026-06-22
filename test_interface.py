"""
Interactive test interface for the multimodal RAG pipeline.

Usage
-----
1. Place your text files in  resources/  (e.g. Engine.txt)
2. Place your images in     resources/  (e.g. engine_schematic.png, photo.jpg)
3. Run this script:  python test_interface.py
4. Type a query at the prompt — the pipeline will search across both
   text chunks and ingested images using CLIP embeddings.

Example session
---------------
    $ python test_interface.py

    [EmbedManager] Loading OpenCLIP on device: cpu
    [EmbedManager] embed_text: 327 chunk(s) in 4.23s → 77.3 chunks/s  (dim=1152, device=cpu)
    [DynamicAgentRAG] add_to_memory: 327 chunks ingested (source_id=43fbacba)
    [EmbedManager] embed_image: 1 image(s) in 0.08s → 12.5 images/s  (dim=1152, device=cpu)
    [DynamicAgentRAG] add_image_to_memory: 1 image ingested (source_id=abc12345, path=resources/photo.jpg)

    Write a search query: > diagram of engine parts
    --- Result 1 (similarity: 0.760, source: 43fbacba) ---
    ...

For programmatic use
--------------------
    from main import DynamicAgentRAG

    rag = DynamicAgentRAG()

    # Ingest a text document
    with open("resources/Engine.txt", "r") as f:
        rag.add_to_memory(f.read())

    # Ingest images in batch (single forward pass through the Vision Transformer)
    rag.add_batch_images_to_memory(["resources/diagram.png", "resources/photo.jpg"])

    # Search
    result = rag.query_memory("exhaust manifold diagram")
    print(rag.format_context(result["context"]))

    # Stats
    print(rag.get_memory_stats())
"""

from main import DynamicAgentRAG

rag = DynamicAgentRAG()

# ------------------------------------------------------------------
# Ingest text documents from resources/
# ------------------------------------------------------------------
print("\n--- Ingesting text documents ---")
try:
    with open("resources/Engine.txt", "r") as file:
        rag.add_to_memory(file.read())
except FileNotFoundError:
    print("  (resources/Engine.txt not found, skipping)")

try:
    with open("resources/Large_language_model.txt", "r") as file:
        rag.add_to_memory(file.read())
except FileNotFoundError:
    print("  (resources/Large_language_model.txt not found, skipping)")

# ------------------------------------------------------------------
# Ingest images from resources/  (add your own image filenames here)
# ------------------------------------------------------------------
print("\n--- Ingesting images ---")
import os

# Common image extensions to auto-discover
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_AUTO_IMAGES = 360

resources_dir = "resources"
if os.path.isdir(resources_dir):
    image_paths = [
        os.path.join(resources_dir, fname)
        for fname in sorted(os.listdir(resources_dir))
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTS
    ]
    if image_paths:
        total_found = len(image_paths)
        if total_found > MAX_AUTO_IMAGES:
            print(
                f"  Found {total_found} images — ingesting first {MAX_AUTO_IMAGES}.\n"
                f"  Edit MAX_AUTO_IMAGES in this script to ingest more."
            )
            image_paths = image_paths[:MAX_AUTO_IMAGES]
        else:
            print(f"  Ingesting {len(image_paths)} images …")
        rag.add_batch_images_to_memory(image_paths)
    else:
        print("  (no image files found in resources/ — add .png or .jpg files)")
else:
    print(f"  (resources/ directory not found)")

# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------
print(f"\n--- Database ready ---")
print(f"  Memories: {rag.get_memory_stats()['total_memories']}")
print(f"  Chunks:   {rag.get_memory_stats()['total_chunks']}")
print()

# ------------------------------------------------------------------
# Interactive query loop
# ------------------------------------------------------------------
while True:
    try:
        txt_in = input("Write a search query: > ")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        break

    result = rag.query_memory(txt_in)
    print(rag.format_context(result["context"]))
