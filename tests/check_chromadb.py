import os
import chromadb # Nadal potrzebne do typów, chociaż nie inicjalizujemy klienta bezpośrednio
from chroma_db_utils import get_chroma_path, get_chroma_collection_and_vector_store # Importujemy funkcje

def check_chromadb_content_for_collection(collection_name: str):
    """
    Sprawdza zawartość bazy danych ChromaDB, wyświetlając liczbę rekordów
    i przykładowe dokumenty z kolekcji ChromaDB.
    """
    CHROMA_PATH = get_chroma_path()
    DB_FILE = os.path.join(CHROMA_PATH, 'chroma.sqlite3')

    if not os.path.exists(DB_FILE):
        print(f"Baza danych {DB_FILE} nie istnieje. Uruchom rag_pipeline.py najpierw, aby ją utworzyć.")
        return

    try:
        # Używamy funkcji z chroma_db_utils, aby uzyskać dostęp do kolekcji ChromaDB
        # Ta funkcja teraz NIE usuwa kolekcji, tylko ją pobiera!
        chroma_collection, _ = get_chroma_collection_and_vector_store(collection_name)

        count_embeddings = chroma_collection.count()
        print(f"\nLiczba osadzeń (chunków) w kolekcji '{chroma_collection.name}': {count_embeddings}")

        if count_embeddings > 0:
            print(f"\n--- Przykładowe dane z kolekcji '{chroma_collection.name}' (pierwsze 3 rekordy) ---")
            results = chroma_collection.get(limit=3, include=['documents', 'metadatas'])
            
            ids = results.get('ids', [])
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])

            for i in range(len(ids)):
                doc_id = ids[i]
                document_text = documents[i]
                metadata = metadatas[i]

                print(f"  ID: {doc_id}")
                print(f"  Dokument (fragment): {document_text[:200]}...")
                print(f"  Metadane: {metadata}")
                print("-" * 30)
        else:
            print(f"Kolekcja '{collection_name}' jest pusta. Dokumenty mogły nie zostać dodane do kolekcji.")

    except Exception as e:
        print(f"Wystąpił błąd podczas sprawdzania zawartości ChromaDB dla kolekcji '{collection_name}': {e}")
        print(f"Upewnij się, że 'rag_pipeline.py' zakończył się pomyślnie i stworzył kolekcję '{collection_name}'.")

    finally:
        pass # PersistentClient zarządza połączeniem

if __name__ == "__main__":
    # Sprawdzamy wszystkie kolekcje
    COLLECTIONS_TO_CHECK = [
        "qwen_embedding_collection",
        "infly_retriever_collection",
        "all_minilm_collection",
    ]
    for col_name in COLLECTIONS_TO_CHECK:
        check_chromadb_content_for_collection(col_name)