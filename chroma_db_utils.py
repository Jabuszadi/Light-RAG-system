import os
import chromadb  
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import CHROMA_PATH

def get_chroma_collection_and_vector_store(db_client: chromadb.PersistentClient, collection_name: str):
    """
    Pobiera lub tworzy kolekcję ChromaDB oraz odpowiadający jej VectorStore.
    """
    chroma_collection = db_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return chroma_collection, vector_store



def get_chroma_path():
    return CHROMA_PATH