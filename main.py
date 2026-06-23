from datetime import datetime
from typing import Any
from chunker import Chunker
from embedder import EmbedManager
from dbmanager import DB

class DynamicAgentRAG():
    def __init__(
            self,
            storage_directory: str = "./vector_db/",
            memory_label: str = "myrag",
            model_name: str = "ViT-B-16-SigLIP",
            pretrained: str = "webli",
    ) -> None:
        self.memory_metadata: dict[str, dict[str, Any]] = {}

        self.db = DB(storage_directory=storage_directory, db_name=memory_label)
        self.embedder = EmbedManager(model_name=model_name, pretrained=pretrained)
        self.chunker = Chunker(chunk_size=950, chunk_overlap=280)

    def add_to_memory(self, text: str) -> str:
        documents, ids, metadatas = self.chunker.chunk_text(text)
        embeddings = self.embedder.embed_text(documents)
        source_id = ids[0].rsplit("_c", 1)[0]

        self.db.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

        self.memory_metadata[source_id] = {
            "chunk_count": len(documents),
            "chunks": ids,
            "timestamp": datetime.now().isoformat(),
            "type": "text",
        }

        return source_id

    def add_image_to_memory(self, image_path: str) -> str:
        """
        Ingest a single image file into the vector database.

        Images are embedded via the vision encoder and stored alongside
        a ``type: "image"`` metadata tag so the retrieval loop can bypass
        paragraph-expansion logic that only applies to text chunks.

        Args:
            image_path:  Absolute or relative path to an image file on disk.

        Returns:
            The generated ``source_id`` (8-char hex digest of the path).
        """
        import hashlib

        # 1. Generate the image embedding
        embedding = self.embedder.embed_image(image_path)[0]

        # 2. Create a unique source ID keyed on the file path
        source_id = hashlib.md5(image_path.encode()).hexdigest()[:8]
        image_id = f"img_{source_id}"

        # 3. Upsert into ChromaDB — store the path as the document
        self.db.collection.upsert(
            ids=[image_id],
            embeddings=[embedding],
            documents=[image_path],
            metadatas=[{
                "type": "image",
                "source_id": source_id,
                "timestamp": datetime.now().isoformat(),
            }],
        )

        # 4. Track in the local metadata registry
        self.memory_metadata[source_id] = {
            "chunk_count": 1,
            "chunks": [image_id],
            "timestamp": datetime.now().isoformat(),
            "type": "image",
        }

        return source_id

    def add_batch_images_to_memory(
        self, image_paths: list[str], batch_size: int = 48,
    ) -> list[str]:
        """
        Ingest multiple images, processing them in sub-batches to keep
        peak GPU/CPU memory bounded while still exploiting vectorised
        forward passes through the vision encoder.

        Args:
            image_paths:  List of paths to image files on disk.
            batch_size:   Max images to encode in a single forward pass.
                          Lower this if you run out of RAM.

        Returns:
            List of generated ``source_id`` strings (one per image).
        """
        import hashlib

        if not image_paths:
            return []

        timestamp = datetime.now().isoformat()
        ids: list[str] = []
        metadatas: list[dict[str, str]] = []
        source_ids: list[str] = []
        all_embeddings: list[list[float]] = []

        # --- Process in sub-batches to keep tensors small ---
        n_total = len(image_paths)
        n_batches = (n_total + batch_size - 1) // batch_size
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_num = start // batch_size + 1
            sub_batch = image_paths[start:end]
            # Simple inline progress percentage
            pct = int(batch_num / n_batches * 100)
            print(f"\r  Embedding images … {pct}%", end="", flush=True)
            sub_embeddings = self.embedder.embed_image(sub_batch)
            all_embeddings.extend(sub_embeddings)
        print()  # newline after progress

        # --- Build upsert payloads ---
        for path in image_paths:
            source_id = hashlib.md5(path.encode()).hexdigest()[:8]
            image_id = f"img_{source_id}"
            source_ids.append(source_id)

            ids.append(image_id)
            metadatas.append({
                "type": "image",
                "source_id": source_id,
                "timestamp": timestamp,
            })

            self.memory_metadata[source_id] = {
                "chunk_count": 1,
                "chunks": [image_id],
                "timestamp": timestamp,
                "type": "image",
            }

        # --- Single ChromaDB upsert with all embeddings ---
        self.db.collection.upsert(
            ids=ids,
            embeddings=all_embeddings,
            documents=image_paths,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

        return source_ids

    def retrieve_context(self, query: str, top_k: int = 3, threshold: float = 0.0) -> list[dict[str, Any]]:
        query_embedding = self.embedder.embed_text(query)[0]
        results = self.db.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )


        if (
            not results or not results["documents"] or (results["documents"][0] is None) or
            not results["distances"] or (results["distances"][0] is None) or
            not results["metadatas"] or (results["metadatas"][0] is None)
        ):
            return []


        context = []
        done_paragraphs = set()

        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance
            if similarity < threshold:
                continue

            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i]

            # --- Image bypass: images have no paragraphs, short-circuit ---
            if metadata.get("type") == "image":
                context.append({
                    "text": f"[IMAGE FILE] {document}",
                    "metadata": metadata,
                    "similarity": similarity,
                    "source_id": metadata.get("source_id"),
                    "chunk_start": 0,
                    "chunk_end": 0,
                })
                continue
            # ----------------------------------------------------------------

            paragraph_id: str = metadata["paragraph_id"]  # type: ignore[assignment]
            source_id: str | None = metadata.get("source_id")  # type: ignore[assignment]

            if paragraph_id in done_paragraphs:
                continue
            done_paragraphs.add(paragraph_id)

            # Fetch and sort all chunks for the matched paragraph
            entries = self._get_paragraph_entries(paragraph_id)
            if not entries:
                continue

            chunk_start: int = entries[0][1]["chunk_index"]  # type: ignore[assignment]
            chunk_end: int = entries[-1][1]["chunk_index"]  # type: ignore[assignment]

            # Expand backwards: prepend the immediately-previous paragraph
            # (the one whose last chunk sits right before our first chunk)
            prev_entries = self._get_adjacent_paragraph(
                source_id, chunk_start - 1, done_paragraphs
            )
            if prev_entries:
                entries = prev_entries + entries
                chunk_start = prev_entries[0][1]["chunk_index"]  # type: ignore[assignment]

            # Expand forwards: append the immediately-next paragraph
            next_entries = self._get_adjacent_paragraph(
                source_id, chunk_end + 1, done_paragraphs
            )
            if next_entries:
                entries = entries + next_entries
                chunk_end = next_entries[-1][1]["chunk_index"]  # type: ignore[assignment]

            ordered_texts = [entry[0] for entry in entries]
            full_text = self.chunker.join_chunks(ordered_texts, self.chunker.chunk_overlap)

            context.append({
                "text": full_text,
                "metadata": metadata,
                "similarity": similarity,
                "source_id": source_id,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
            })

        return context

    def _get_paragraph_entries(self, paragraph_id: str) -> list[tuple[Any, Any]]:
        """Return (text, metadata) pairs for every chunk of *paragraph_id*,
        sorted by chunk_index.  Returns an empty list when nothing is found."""
        result = self.db.collection.get(
            where={"paragraph_id": paragraph_id},   # type: ignore[arg-type]
            include=["documents", "metadatas"],
        )
        docs = result["documents"]
        metas = result["metadatas"]
        if not (docs and metas):
            return []
        entries = list(zip(docs, metas))
        entries.sort(key=lambda e: e[1]["chunk_index"])  # type: ignore[arg-type]
        return entries

    def _get_adjacent_paragraph(
        self, source_id: str | None, target_chunk_index: int,
        done_paragraphs: set[str],
    ) -> list[tuple[Any, Any]] | None:
        """If the chunk at *target_chunk_index* exists and belongs to a
        paragraph we haven't seen yet, return that paragraph's entries
        (sorted) and mark it seen.  Otherwise return None."""
        if source_id is None or target_chunk_index < 0:
            return None

        probe = self.db.collection.get(
            where={"$and": [                              # type: ignore[arg-type]
                {"source_id": source_id},
                {"chunk_index": target_chunk_index},
            ]},
            include=["metadatas"],
        )
        if not (probe["metadatas"] and probe["metadatas"][0] is not None):
            return None

        neighbour_id: str = probe["metadatas"][0]["paragraph_id"]  # type: ignore[assignment]
        if neighbour_id in done_paragraphs:
            return None
        done_paragraphs.add(neighbour_id)

        return self._get_paragraph_entries(neighbour_id)

    def forget_memory(self, source_id: str) -> None:
        if source_id in self.memory_metadata:
            chunks = self.memory_metadata[source_id]["chunks"]
            self.db.collection.delete(ids=chunks)
            del self.memory_metadata[source_id]

    def query_memory(self, query: str) -> dict[str, Any]:
        context = self.retrieve_context(query)
        return {
            "query": query,
            "context": context,
            "num_retrieved": len(context),
            "sources": list(set([c["source_id"] for c in context]))
        }

    def format_context(self, context: dict) -> str:
        """Return context as human-readable text, one section per result."""
        if not context:
            return "(no results found)"

        lines = []
        for i, item in enumerate(context):
            lines.append(
                f"--- Result {i + 1} "
                f"(similarity: {item['similarity']:.3f}, "
                f"source: {item.get('source_id', 'unknown')}) ---"
            )
            lines.append(item["text"])
            lines.append("")

        return "\n".join(lines)

    def get_memory_stats(self) -> dict[str, int]:
        return {
            "total_memories": len(self.memory_metadata),
            "total_chunks": self.db.collection.count(),
        }


class ContextAwareRAG(DynamicAgentRAG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
        self.context_window = []

    def add_context(self, context_item, importance=0.5):
        self.context_window.append({
            "text": context_item,
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.context_window) > 10:
            self.context_window.pop(0)

    def get_augmented_query(self, query):
        if not self.context_window:
            return query
        recent = [ctx["text"] for ctx in self.context_window[-3:] if ctx["importance"] > 0.3]
        if recent:
            return f"{query} [Context: {' '.join(recent)}]"
        return query

    def intelligent_retrieve(self, query, use_context=True):
        search_query = self.get_augmented_query(query) if use_context else query
        exact = self.retrieve_context(search_query, top_k=3)
        broad = self.retrieve_context(query, top_k=2, threshold=0.3)
        # ... deduplicate and sort by similarity