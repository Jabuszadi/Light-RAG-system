import os
from llama_index.core.prompts.prompts import SimpleInputPrompt

LLM_MODEL = "qwen3:8b" 

# Definicja ścieżki do ChromaDB
CHROMA_PATH = os.path.join(os.getcwd(), 'chroma_db')

# Konfiguracja kolekcji i odpowiadających im modeli osadzania
COLLECTIONS_CONFIG = [
    # {"name": "qwen_embedding_collection", "embedding_model": "Qwen/Qwen3-Embedding-4B"},
    {"name": "infly_retriever_collection", "embedding_model": "infly/inf-retriever-v1-1.5b"},
    {"name": "all_minilm_collection", "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "paraphrase_multilingual_collection", "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"},
    {"name": "all_roberta_large_v1_collection", "embedding_model": "sentence-transformers/all-roberta-large-v1"},
]

# Definicja promptu
CUSTOM_QUERY_PROMPT = SimpleInputPrompt(
    """Jesteś asystentem AI, którego jedynym zadaniem jest odpowiadanie na pytania, korzystając **WYŁĄCZNIE** z dostarczonych poniżej fragmentów dokumentów (kontekstu).

---------------------
{context_str}
---------------------

Biorąc pod uwagę ten kontekst:
1. Odpowiedz na pytanie: {query_str}
2. Twoja odpowiedź musi być w pełni poparta dostarczonymi fragmentami.
3. **NIE dodawaj żadnych informacji spoza dostarczonego kontekstu, nawet jeśli znasz odpowiedź.**
4. Jeśli odpowiedź na pytanie **NIE znajduje się w dostarczonych fragmentach**, odpowiedz **dokładnie**: "Przepraszam, nie jestem w stanie znaleźć odpowiedzi na to pytanie w dostarczonych dokumentach."
5. Jeśli pytanie wymaga konkretnych nazw modeli, liczb, czy wymagań VRAM, wyodrębnij je precyzyjnie z kontekstu.

Odpowiedź: """
)

STATIC_QUESTIONS = [
    "Kim jest Sam Altman?",
    "Jakie modele LLaMa są dostępne?",
    "Kto stworzył PLLuM?",
    "Jaki model najlepiej działa na GPU z 24 GB VRAM?",
    "Ile pamięci potrzeba, aby odpalić najmniejszy model llm? Podaj ilość VRAM i podaj jego nazwę"
]

# Modele LLM do przetestowania (Ollama)
LLM_MODELS_TO_TEST = [

    # "gemma3:270m",
    # "qwen3:0.6b",
    # "llama3.2:3b",
    # "deepseek-r1:7b",
    # "llama3.1:8b",
    # "antoniprzybylik/llama-pllum:8b",
    "qwen3:8b",
    "glm4:9b"
    # "mistral-small3.2",
    # "gpt-oss:20b",
    # "gemma3:27b"

]

GROUND_TRUTH_DATA = {
    "Kim jest Sam Altman?": {
        "answer": "Przepraszam, nie jestem w stanie znaleźć odpowiedzi na to pytanie w dostarczonych dokumentach.",
        "contexts": "Brak danych o Samie Altmanie w dostarczonych dokumentach.",
    },
    "Jakie modele LLaMa są dostępne?": {
        "answer": """Dostępne modele LLaMA to:
    LLaMA 1: 7 B, 13 B, 65 B parametrów.
    LLaMA 2: 7 B, 13 B, 70 B parametrów.
    Code Llama (wariant LLaMA 2): 7 B, 13 B, 30/33/34 B parametrów.
    LLaMA 3: 8 B, 70 B parametrów.
    LLaMA 3.1: 8 B, 70 B, 405 B parametrów.
    LLaMA 3.2: 1 B, 3 B (tekstowe) oraz Llama 3.2-Vision 11 B, 90 B (multimodalne).
    LLaMA 4: Llama 4 Scout, Llama 4 Maverick, oraz zapowiedziana wersja Llama 4 Behemoth.""",
        "contexts": """LLaMA 1 była pierwszą publicznie dostępną rodziną modeli Mety. Obejmuje warianty **7 B**, **13 B** i **65 B** parametrów.
LLaMA 2 wprowadziła komercyjnie dostępne modele **7 B**, **13 B** i **70 B**.
Code Llama to wariant LLaMA 2 specjalizujący się w generacji kodu i obsłudze zadań programistycznych. Modele są dostępne w rozmiarach 7 B, 13 B oraz 30/33/34 B i działają z długim kontekstem (do 100 k tokenów).
Dostępne modele to **8 B** i **70 B**.
Wersja 3.1 wprowadziła modele **8 B**, **70 B** i **405 B**, rozszerzając kontekst do 128 k tokenów i poprawiając stabilność.
Ta aktualizacja rozszerzyła rodzinę o lekkie modele **1 B** i **3 B** (tekstowe) oraz warianty multimodalne **Llama 3.2-Vision 11 B** i **90 B**.
Czwarta generacja wprowadza architekturę **Mixture of Experts (MoE)** oraz natywne przetwarzanie multimodalne. Pierwsze modele to **Llama 4 Scout** i **Llama 4 Maverick**, a w zapowiedzi jest wersja **Llama 4 Behemoth**.""",
    },
    "Kto stworzył PLLuM?": {
        "answer": "PLLuM został stworzony przez konsorcjum polskich uczelni i instytutów, koordynowane przez Politechnikę Wrocławską i wspierane m.in. przez NASK PIB, Instytut Podstaw Informatyki PAN, Ośrodek Przetwarzania Informacji PIB, Uniwersytet Łódzki oraz Instytut Slawistyki PAN.",
        "contexts": "PLLuM (Polish Large Language Model) jest projektem konsorcjum polskich uczelni i instytutów, koordynowanym przez Politechnikę Wrocławską i wspieranym m.in. przez NASK PIB, Instytut Podstaw Informatyki PAN, Ośrodek Przetwarzania Informacji PIB, Uniwersytet Łódzki oraz Instytut Slawistyki PANhttps://primotly.com/article/pllum-polish-large-language-model-artificial-intelligence#:~:text=PLLuM%20was%20officially%20presented%20on,linguistic%20accuracy%20and%20thematic%20diversity.",
    },
    "Jaki model najlepiej działa na GPU z 24 GB VRAM?": {
        "answer": """Na GPU z 24 GB VRAM najlepiej działa:
    Mistral NeMo 12B (w wersji fp8 wymaga 16 GB VRAM, więc 24 GB jest wystarczające, a model oferuje 128k kontekstu i jest następcą Mistral 7B).
    Ministral 8B (wymaga 24 GB VRAM, jest kompaktowy i przeznaczony do zastosowań on-device/edge).
    LLaMA 3.1 8B (w wersji FP16 wymaga ok. 16 GB VRAM, a model ma kontekst do 128 tys. tokenów).
    Mixtral 8x7B w wersji 4-bitowej kwantyzacji (wymaga ~22.5 GB VRAM).
    LLaMA 3 70B (kwantyzacja 4-bitowa pozwala uruchomić go na pojedynczym GPU RTX 4090 z 24 GB VRAM, ale kosztem wydajności).""",
        "contexts": """| Mistral NeMo 12B | 12 B | do 128k tokens | 28 GB (bf16) / 16 GB (fp8)https://docs.mistral.ai/getting-started/models/weights/#:~:text=Mistral,fp8 | Apache 2.0 | wielojęzyczny model tekstowy z Tekken tokenizerem; zamiennik Mistral 7B |
| Ministral 8B | 8 B | 128k tokens (32k w vLLM) | 24 GB | Mistral Commercial/Research License | niskolatencyjne asystenty on‑device |
* **8 B** – w precyzji FP16 wymaga ok. 16 GB VRAM, w FP8 8 GB, a w INT4 4 GBhttps://huggingface.co/blog/llama31#:~:text=Model%20Size%20FP16%20FP8%20INT4,405%20GB%20%20203%20GB.
| Mixtral 8×7B | 46.7 B (12.9 B aktywne) | 32k tokens | 100 GB | Apache 2.0 | wydajna mieszanka ekspertów (MoE) do długich odpowiedzi, RAG |
Dyskusje na HuggingFace podają, że wersja 4‑bitowa wymaga ~22.5 GB VRAM, 8‑bitowa ~45 GB, a pełna półprecyzja ~90 GBhttps://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/discussions/3#:~:text=Quick%20math%20,parameters.
Artykuł firmy Picovoice podaje, że model **Llama 3 70 B** w pełnej precyzji potrzebuje **ponad 140 GB VRAM**, a specjalne techniki kwantyzacji (np. 4‑bit) pozwalają uruchomić go na pojedynczym GPU RTX 4090 (24 GB) kosztem wydajnościhttps://picovoice.ai/blog/unleash-the-power-of-llama3-70b-on-your-everyday-computer/#:~:text=family%20ai,uncommon%20and%20can%20be%20expensive.""",
    },
    "Ile pamięci potrzeba, aby odpalić najmniejszy model llm? Podaj ilość VRAM i podaj jego nazwę": {
        "answer": "Najmniejszy model LLM wymieniony w dokumentach, który podaje konkretne wymagania pamięci, to LLaMA 3.2 (model 1B tekstowy), który potrzebuje 0,75 GB VRAM w kwantyzacji INT4.",
        "contexts": """* **Sprzęt** – artykuł Hugging Face podaje, że **model 3 B** wymaga **ok. 6,5 GB VRAM** w bf16/fp16, **3,2 GB** w FP8 i **1,75 GB** w INT4; **model 1 B** potrzebuje **2,5 GB** (FP16), **1,25 GB** (FP8) lub **0,75 GB** (INT4)https://huggingface.co/blog/llama32#:~:text=Llama%203,Language%20Models.""",
    },
}