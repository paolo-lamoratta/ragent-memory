import hashlib
from datetime import datetime
from typing import Any


class Chunker():
    def __init__(self, chunk_size: int = 950, chunk_overlap: int = 280) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def chunk_text(self, text: str) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        source_id = hashlib.md5(text.encode()).hexdigest()[:8]

        documents = []
        ids = []
        metadatas = []

        start = 0
        text_len = len(text)
        paragraph_id = None
        start_new_paragraph = True   # first chunk always starts a paragraph

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            ends_at_paragraph = False

            # Try to cut cleanly at a sentence boundary (skip for the final chunk)
            if end < text_len:
                chunk_slice = text[start:end]
                best_sep_idx = -1
                best_sep_is_paragraph = False

                for sep in ['.\n\n', '.\n']:
                    idx = chunk_slice.rfind(sep)
                    if idx > best_sep_idx:
                        best_sep_idx = idx
                        best_sep_is_paragraph = (sep == '.\n\n')

                # Only adjust when the boundary sits in the back half of the
                # chunk — otherwise we create tiny fragments.  Paragraph
                # boundaries naturally cluster at the end of sentences, so
                # they are almost always in the back half for realistic sizes.
                if best_sep_idx > (self.chunk_size * 0.5):
                    end = start + best_sep_idx + 1
                    ends_at_paragraph = best_sep_is_paragraph

            chunk_text_str = text[start:end].strip()

            if start_new_paragraph:
                paragraph_id = hashlib.md5(chunk_text_str.encode()).hexdigest()[:8]
                start_new_paragraph = False

            if chunk_text_str:
                chunk_id = f"{source_id}_c{len(documents)}"
                documents.append(chunk_text_str)
                ids.append(chunk_id)
                metadatas.append({
                    "chunk_index": len(documents),
                    "start_char": start,
                    "end_char": end,
                    "timestamp": datetime.now().isoformat(),
                    "paragraph_id": paragraph_id,
                    "source_id": source_id,
                })

            # Next chunk starts a new paragraph only when this one
            # ended at a paragraph boundary.
            start_new_paragraph = ends_at_paragraph

            # If this chunk reached the end of the text we are done —
            # otherwise the overlap would create near-duplicate fragments.
            if end >= text_len:
                break

            start = max(start + 1, end - self.chunk_overlap)

        return documents, ids, metadatas


    def join_chunks(self, chunks : list[str], overlap : int) -> str:
        joined = ""
        for chunk in chunks:
            if (not joined):
                joined += chunk
            else:
                joined += chunk[overlap:]

        return joined
