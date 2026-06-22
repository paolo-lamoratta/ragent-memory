from sentence_transformers import SentenceTransformer


class EmbedManager():
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        
        
        self.client = SentenceTransformer(embedding_model, local_files_only=True)
        self.embedding_dim = self.client.get_embedding_dimension()

    def embed(self, text: str | list[str]) -> list[float]:
        if isinstance(text, str):
            return self.client.encode(text).tolist()

        elif isinstance(text, list):
            return self.client.encode(text, batch_size=32).tolist()
        
if __name__ == "__main__":
    client = EmbedManager()
    print("Hi everybody, this is the embedding vector of this string\n\n")
    print(client.embed("Hi everybody, this is the embedding vector of this string"))