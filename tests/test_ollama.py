import os
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import sys # Nowy import dla sys.stdout.flush()

def test_ollama_model_streaming():
    LLM_MODEL = "mistral-small3.2:latest"
    TEST_QUERY = "Opisz krótko, co to jest rekurencja w programowaniu."
    TIMEOUT_SECONDS = 600.0 # Zwiększony timeout na wszelki wypadek

    print(f"--- Testowanie modelu Ollama (Streaming): {LLM_MODEL} ---")
    print(f"Upewnij się, że serwer Ollama jest uruchomiony i model '{LLM_MODEL}' jest pobrany (ollama pull {LLM_MODEL}).")
    print(f"Próba zapytania: '{TEST_QUERY}' (streaming)")

    try:
        llm = Ollama(model=LLM_MODEL, request_timeout=TIMEOUT_SECONDS)
        Settings.llm = llm

        print("\nOdpowiedź (streaming): ")
        full_response_text = ""
        # --- Zmiana tutaj: używamy stream_complete zamiast complete ---
        streaming_response = llm.stream_complete(TEST_QUERY)
        
        for text_chunk in streaming_response:
            print(text_chunk.delta, end="") # Używamy .delta dla streamingu
            full_response_text += text_chunk.delta
            sys.stdout.flush() # Wymusza natychmiastowe wyświetlanie w konsoli
        print() # Nowa linia po zakończeniu streamowania
        
        print("\n--- Model Ollama odpowiedział (streaming)! ---")
        print("Model działa poprawnie. Możesz teraz spróbować uruchomić chat.py z włączonym streamingiem.")

    except Exception as e:
        print(f"\n--- BŁĄD PODCZAS TESTU MODELU OLLAMA (Streaming) ---")
        print(f"Wystąpił błąd: {e}")
        print(f"Najczęstsze przyczyny:")
        print(f"1. Serwer Ollama nie jest uruchomiony (sprawdź w tle lub uruchom 'ollama serve' w terminalu).")
        print(f"2. Model '{LLM_MODEL}' nie jest pobrany (pobierz go komendą 'ollama pull {LLM_MODEL}').")
        print(f"3. Model jest zbyt duży/wolny dla Twojego sprzętu i przekracza limit czasu ({TIMEOUT_SECONDS} sekund).")
        print(f"   Spróbuj zwiększyć TIMEOUT_SECONDS lub użyć mniejszego modelu, np. 'llama2' lub 'mistral'.")

if __name__ == "__main__":
    test_ollama_model_streaming() # Zmieniono wywołanie funkcji