import chromadb
from chromadb.config import Settings

class DB():
    def __init__(self, storage_directory: str = "./vector_db/", db_name: str = "myrag") -> None:
        self.client = chromadb.PersistentClient(
            path=storage_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=db_name,
            metadata={"hnsw:space": "cosine"}
        )