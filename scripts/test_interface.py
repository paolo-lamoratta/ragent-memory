"""Interactive test shell for the multimodal RAG pipeline."""

from ragent_memory import DynamicAgentRAG

rag = DynamicAgentRAG()

# --- Ingest text documents --------------------------------------------------
import os

for fname in sorted(os.listdir("resources")):
    if fname.endswith(".txt"):
        with open(os.path.join("resources", fname)) as f:
            rag.add_to_memory(f.read())

# --- Ingest images ----------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_IMAGES = 2000

image_paths = [
    os.path.join("resources", f)
    for f in sorted(os.listdir("resources"))
    if os.path.splitext(f)[1].lower() in IMAGE_EXTS
]
if image_paths:
    rag.add_batch_images_to_memory(image_paths[:MAX_IMAGES])

# --- Query loop -------------------------------------------------------------
while True:
    try:
        q = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    result = rag.query_memory(q)
    print(rag.format_context(result["context"]))
