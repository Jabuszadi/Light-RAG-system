import os
import shutil
from dotenv import load_dotenv
import chromadb 
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from chroma_db_utils import get_chroma_collection_and_vector_store 
from config import CHROMA_PATH, COLLECTIONS_CONFIG, LLM_MODEL  

# Załadowanie zmiennych środowiskowych z pliku .env (np. HF_TOKEN)
load_dotenv()

# Konfiguracja Ollamy (dla LLM, który będzie używany do syntezy odpowiedzi)

Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)  
# --- CAŁKOWITE CZYSZCZENIE FOLDERU CHROMA_DB ---
if os.path.exists(CHROMA_PATH):
    print(f"\n--- Usuwanie istniejącego katalogu ChromaDB: {CHROMA_PATH} ---")
    try:
        shutil.rmtree(CHROMA_PATH)
        print("Katalog usunięty pomyślnie.")
    except Exception as e:
        print(f"Błąd podczas usuwania katalogu ChromaDB: {e}")
        print("Proszę upewnić się, że żadne inne procesy nie korzystają z tego katalogu.")
print("--- Zakończono czyszczenie ---\n")

os.makedirs(CHROMA_PATH, exist_ok=True)
db = chromadb.PersistentClient(path=CHROMA_PATH)  

for config in COLLECTIONS_CONFIG:
    collection_name = config["name"]
    embedding_model_name = config["embedding_model"]

    print(f"\n--- Przetwarzam kolekcję: {collection_name} z modelem osadzania: {embedding_model_name} ---")

    try:
        embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            
        )
        Settings.embed_model = embed_model  

        # Przekazujemy zainicjalizowany obiekt 'db' ===
        chroma_collection, vector_store = get_chroma_collection_and_vector_store(db, collection_name)

        total_nodes = chroma_collection.count()
        if total_nodes == 0:
            print("Kolekcja ChromaDB jest pusta. Tworzę indeks i dodaję dokumenty...")

            documents = SimpleDirectoryReader(input_dir="docs").load_data()
            parser = MarkdownNodeParser()
            nodes = parser.get_nodes_from_documents(documents)

            print(f"Utworzono {len(nodes)} węzłów (nodes) z dokumentów.")

            for i, node in enumerate(nodes):
                node_embedding = embed_model.get_text_embedding(node.text)
                node.embedding = node_embedding
                if i % 20 == 0:
                    print(f"  Wygenerowano embedding dla węzła {i}/{len(nodes)}")

            vector_store.add(nodes)
            print("Dokumenty dodane do ChromaDB.")

            # Potwierdzenie liczby osadzeń po dodaniu
            count_after_add = chroma_collection.count()
            print(f"Potwierdzenie: Liczba osadzeń w kolekcji '{collection_name}' po dodaniu: {count_after_add}")
        else:
            print(f"Kolekcja ChromaDB '{collection_name}' zawiera już {total_nodes} osadzeń. Pomiętanie indeksowania.")

        print(f"Indeks LlamaIndex dla kolekcji '{collection_name}' gotowy.")

    except Exception as e:
        print(f"Błąd podczas przetwarzania kolekcji '{collection_name}': {e}")
        print("Upewnij się, że serwer Ollama jest uruchomiony i model osadzania jest dostępny.")
        continue

print("\n--- Zakończono indeksowanie wszystkich kolekcji ---\n")