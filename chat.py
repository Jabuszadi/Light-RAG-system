import chromadb
from chroma_db_utils import get_chroma_path, get_chroma_collection_and_vector_store
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from config import LLM_MODEL, COLLECTIONS_CONFIG, CUSTOM_QUERY_PROMPT
from llama_index.llms.ollama import Ollama

SELECTED_EMBEDDING_COLLECTION_NAME = "infly_retriever_collection"

def chat_with_rag():
    print(f"Skonfigurowano LLM: {LLM_MODEL} dla czatu.")

    llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
    Settings.llm = llm

    # Inicjalizacja klienta ChromaDB
    db = chromadb.PersistentClient(path=get_chroma_path())

    query_engines_by_collection = {}

    # Wyszukujemy tylko wybraną kolekcję
    selected_config = next((config for config in COLLECTIONS_CONFIG if config["name"] == SELECTED_EMBEDDING_COLLECTION_NAME), None)

    if not selected_config:
        print(f"Błąd: Nie znaleziono konfiguracji dla kolekcji '{SELECTED_EMBEDDING_COLLECTION_NAME}' w config.py.")
        return

    collection_name = selected_config["name"]
    embedding_model_name = selected_config["embedding_model"]
    print(f"  Ładuję wybraną kolekcję: {collection_name} z modelem osadzania: {embedding_model_name}")

    try:
        # Inicjalizacja ChromaDB i Vector Store dla wybranej kolekcji
        chroma_collection, vector_store = get_chroma_collection_and_vector_store(db, collection_name)

        # Ustawienie modelu embeddingowego dla Settings LlamaIndex
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

        index = VectorStoreIndex.from_vector_store(vector_store)

        # Konfiguracja Retrievera
        retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

        # Konfiguracja Response Synthesizera
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="compact",
            text_qa_template=CUSTOM_QUERY_PROMPT,  # Używamy niestandardowego promptu
        )

        # Tworzenie QueryEngine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        query_engines_by_collection[collection_name] = query_engine
        print(f"  Komponenty dla kolekcji '{collection_name}' gotowe. Liczba osadzeń: {chroma_collection.count()}")

    except Exception as e:
        print(f"  Błąd podczas ładowania kolekcji '{collection_name}': {e}")
        print(f"  Upewnij się, że 'rag_pipeline.py' zakończył się pomyślnie i stworzył kolekcję '{collection_name}'.")

    if not query_engines_by_collection:
        print("Nie udało się załadować komponentów indeksów dla wybranej kolekcji. Nie można uruchomić czatu RAG.")
        return

    print("\n--- Tryb interaktywny RAG ---")
    print(f"Używam kolekcji: {SELECTED_EMBEDDING_COLLECTION_NAME}")
    print("Wpisz 'wyjdz' aby zakończyć.")

    active_query_engine = list(query_engines_by_collection.values())[0]

    while True:
        user_question = input("Twoje pytanie: ")
        if user_question.lower() == 'wyjdz':
            break

        if not user_question.strip():
            print("Pytanie nie może być puste. Spróbuj ponownie.")
            continue

        try:
            print(f"\n--- Odpowiedź z kolekcji: {SELECTED_EMBEDDING_COLLECTION_NAME} ---")
            response = active_query_engine.query(user_question)
            print(response.response)

        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania zapytania w kolekcji '{SELECTED_EMBEDDING_COLLECTION_NAME}': {e}")
            print(f"Upewnij się, że serwer Ollama jest uruchomiony i model LLM ('{LLM_MODEL}') jest pobrany.")


if __name__ == "__main__":
    chat_with_rag()