import hashlib
from datetime import datetime
from typing import Any


class Chunker():
    def __init__(self, chunk_size: int = 950, chunk_overlap: int = 280) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def chunk_text(self, text: str) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        # If no source_id was passed in, generate one from the whole text
        source_id = hashlib.md5(text.encode()).hexdigest()[:8]

        documents = []
        ids = []
        metadatas = []

        start = 0
        text_length = len(text)
        paragraph_id = None
        is_last = True
        is_first = True

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            if end < text_length:
                chunk_slice = text[start:end]
                best_sep_idx = -1
                if (is_last == True):
                    is_first = True
                    is_last = False
                else: 
                    is_first = False

                for sep in ['.\n\n', '.\n']:
                    idx = chunk_slice.rfind(sep)
                    if idx > best_sep_idx:
                        best_sep_idx = idx
                        is_last = True

                
                if best_sep_idx > (self.chunk_size * 0.5):
                    end = start + best_sep_idx + 1

            chunk_text_str = text[start:end].strip()
            
            if (is_first):
                paragraph_id = hashlib.md5(chunk_text_str.encode()).hexdigest()[:8]


            if chunk_text_str:
                # BULLETPROOF ID: e.g. "a1b2c3d4_c0", "a1b2c3d4_c1"
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
