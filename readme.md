# Projekt RAG (Retrieval-Augmented Generation)

## Opis Projektu
Ten projekt implementuje system Retrieval-Augmented Generation (RAG) z wykorzystaniem LlamaIndex, Ollamy i ChromaDB. Celem projektu jest demonstracja działania systemów RAG, ewaluacja różnych modeli językowych (LLM) i modeli embeddingowych, oraz interaktywny czat.

## Funkcjonalności
- **Budowanie Bazy Wiedzy (ChromaDB)**: Proces pobierania dokumentów (z katalogu `docs/`), dzielenia ich na fragmenty (chunks) i osadzania ich w wektorowej bazie danych ChromaDB.
- **Interaktywny Czat RAG**: Aplikacja do prowadzenia rozmów z modelem LLM, który odpowiada na pytania, korzystając z informacji odzyskanych z bazy ChromaDB.
- **Ewaluacja Modeli LLM i Embeddingowych**: Skrypt do kompleksowej oceny wydajności różnych konfiguracji LLM i modeli embeddingowych, z wykorzystaniem metryk LlamaIndex (Faithfulness, Answer Relevancy, Context Relevancy, Correctness) oraz metryk retriewala (Precision, Recall, MRR).
- **Powtarzalna Ewaluacja**: Skrypt do wielokrotnego uruchamiania ewaluacji w celu sprawdzenia stabilności i powtarzalności wyników.

## Wymagania
- Python 3.10+
- Ollama (lokalny serwer LLM)
- Zalecane środowisko wirtualne (conda lub venv)
- Dodatkowa pamięć wirtualna/plik stronicowania (szczególnie dla większych modeli LLM, np. 9B i większych)

## Użycie

### 1. Interaktywny Czat RAG
Uruchom aplikację czatu, która używa wybranej kolekcji embeddingów (domyślnie `infly_retriever_collection` - można zmienić w `chat.py`).
```bash
python chat.py
```
Po uruchomieniu możesz zadawać pytania modelowi. Wpisz `wyjdz`, aby zakończyć.

### 2. Ewaluacja Modeli
Ten skrypt uruchomi kompleksową ewaluację wszystkich modeli LLM zdefiniowanych w `LLM_MODELS_TO_TEST` i wszystkich kolekcji embeddingów z `COLLECTIONS_CONFIG` w `config.py`.
Wyniki zostaną zapisane w katalogu `evaluation_results/`.
```bash
python evaluate_models.py
```

### 3. Powtarzalna Ewaluacja
Ten skrypt pozwala na wielokrotne uruchomienie ewaluacji i agregację wyników.

**Uruchomienie pełnych powtórzeń (domyślnie 10):**
Tworzy katalog `evaluation_results_repetitive/`, a w nim podkatalogi `run_01`, `run_02` itd.
```bash
python repetitive_evaluate.py
```


## Struktura Projektu
-   `chat.py`: Skrypt do interaktywnego czatu RAG. Umożliwia zadawanie pytań i otrzymywanie odpowiedzi wzbogaconych o informacje z bazy wiedzy.
-   `config.py`: Plik konfiguracyjny, zawierający globalne ustawienia dla projektu, takie jak:
    *   Definicje modeli LLM do testowania.
    *   Konfiguracje kolekcji embeddingów.
    *   Niestandardowe prompty dla modeli.
    *   Dane `ground_truth` (referencyjne pytania i odpowiedzi) do ewaluacji.
-   `chroma_db_utils.py`: Zbiór funkcji pomocniczych do interakcji z bazą danych ChromaDB. Obejmuje funkcje do uzyskiwania kolekcji i VectorStore.
-   `docs/`: Katalog przeznaczony do przechowywania dokumentów źródłowych (np. plików `.txt`), które są indeksowane i używane do budowy bazy wiedzy.
-   `tests/`: Katalog zawierający skrypty testowe i narzędzia do weryfikacji komponentów projektu:
    *   `check_chromadb.py`: Skrypt do sprawdzania i weryfikacji działania bazy danych ChromaDB.
    *   `test_ollama.py`: Skrypt zawierający testy dla integracji z modelem Ollama.
-   `evaluate_models.py`: Skrypt odpowiedzialny za przeprowadzanie kompleksowej ewaluacji różnych modeli RAG. Ocenia wierność (faithfulness), trafność odpowiedzi (answer relevancy), trafność kontekstu (context relevancy) i poprawność (correctness) generowanych odpowiedzi.
-   `repetitive_evaluate.py`: Skrypt przeznaczony do automatyzowania powtarzalnych cykli ewaluacji modeli RAG i agregacji wyników, co ułatwia analizę stabilności i wydajności modeli.
-   `data_ingestion.py`: Skrypt do inicjalizacji, budowania i aktualizacji bazy danych ChromaDB na podstawie dokumentów znajdujących się w katalogu `docs/`.
-   `chroma_db/`: Katalog, w którym ChromaDB przechowuje wszystkie dane wektorowe i metadane kolekcji.


## Konfiguracja (`config.py`)
W pliku `config.py` możesz dostosować:
- `LLM_MODEL`: Domyślny model LLM używany w czacie.
- `CHROMA_PATH`: Ścieżka do bazy danych ChromaDB.
- `COLLECTIONS_CONFIG`: Listę słowników, z których każdy definiuje nazwę kolekcji ChromaDB (`"name"`) i odpowiadający jej model embeddingowy (`"embedding_model"`).
- `CUSTOM_QUERY_PROMPT`: Szablon promptu dla modelu LLM.
- `STATIC_QUESTIONS`: Listę pytań używanych do ewaluacji.
- `LLM_MODELS_TO_TEST`: Listę modeli LLM do przetestowania w ewaluacji.
- `GROUND_TRUTH_DATA`: Zbiór danych referencyjnych (pytanie, poprawna odpowiedź, konteksty) do ewaluacji.

## Przetestowane modele emmeddingowe:
Qwen/Qwen3-Embedding-4B
infly/inf-retriever-v1-1.5b
sentence-transformers/all-MiniLM-L6-v2
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
sentence-transformers/all-roberta-large-v1

## Przetestowane modele LLM:
- "gemma3:270m",
- "qwen3:0.6b",
- "llama3.2:3b",
- "deepseek-r1:7b",
- "llama3.1:8b",
- "antoniprzybylik/llama-pllum:8b",
- "qwen3:8b",
- "glm4:9b"
- "mistral-small3.2",
- "gpt-oss:20b",
- "gemma3:27b"

## Obserwacje
Po przeprowadzeniu wstępnych testów, większośc wyżej zaprezentowanych modeli nie spełniło oczekiwań albo 
małe modele ignorowały polecenie trzymania sie danych zamieszczonych w folderze docs 
(Najlepiej widoczne w przypadku dodanego zapytania "Kim jest Sam Altman?") albo modele nie udzielały żadnej 
odp.ze względu na ich ograniczone zdolności "rozumowania" i podążania za złożonymi instrukcjami.
(Pytanie: "Ile pamięci potrzeba, aby odpalić najmniejszy model llm? Podaj ilość VRAM i podaj jego nazwę") 
Z kolei modele o większej ilości parametrów niż 8B często nie odpowiadały w wyznaczonym czasie (300 sek). 
Doszedł też problem stabilności być może ze względu na za małą pamięć
VRAM (24GB), modele powyzej 9B parametrów sprawiały, że programowi zabrakło pamięci do działania i doprowadzały
do jego zamknięcia. Po wytypypowaniu kilku najlepiej rokujących modeli przeszedłem do testu ich repetywnośći
oraz ewalauacji za pomocą innego modelu oLLM. Wyniki można przejrzeć w folderze old oraz evaluation_results.
Faworytami okazli się glm4:9b oraz qwen3:8b. Jednak po serii repetytywnych testów, aby sprawdzić ich
stabilność odpowiedzi i działania (wyniki w folderze: "evaluation_results_repetitive"). 
GLM4:9b, okazał się nie być tak stabilny jak Qwen3:8b. Więc pozostał na arenie jedynie ten model. 
Na tym testy zostały zakończone. 

## Wnioski
Qwen3:8b okazał się najlpesyzm modelem ze względu na równowagę między wydajnością, stabilnością 
i zdolnością do podążania za instrukcjami. 

Jeżeli chodzi z kolei o najlepiej dopasowane modele embeddingowe z przetestowanych tutaj 
To zostali wytypowani dwaj faworyci. Jeżeli zależy nam na najszybszej odpowiedzi to warto skorzystać z
all-MiniLM-L6-v2 dla qwen3:8b. Jest to dobry wybór, jeśli priorytetem jest szybkość  (średnio 20.40s)
i wierność kontekstowi ma najwyższą avg_faithfulness (0.77) dla qwen3:8b.
Jednak kosztem minimalnie niższej trafności odpowiedzi/kontekstu.
 
Drugim embeddem z kolei wartym uwagi będzie model inf-retriever-v1-1.5b,  który osiągnął najwyższy wynik avg_correctness (3.97) 
dla qwen3:8b. Jest również przyzwoity pod względem avg_answer_relevancy (0.97) i ma akceptowalny czas odpowiedzi.
Nie wymaga też takich zasobów jakie Qwen3-Embedding-4B. Niestety infly ma najniższą avg_faithfulness (0.63) i 
avg_context_relevancy (0.81) w porównaniu do innych czołowych modeli embeddingowych, co sugeruje, 
że odzyskane konteksty mogą być mniej trafne, a odpowiedzi mniej ściśle poparte źródłami.
