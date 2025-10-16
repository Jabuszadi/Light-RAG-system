import os
import time
import json
import sys
import shutil  
import pandas as pd
import datetime  
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import Settings, VectorStoreIndex, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
import llama_index.core.evaluation as evaluation
from llama_index.core.evaluation import CorrectnessEvaluator
from config import CHROMA_PATH, COLLECTIONS_CONFIG, STATIC_QUESTIONS, LLM_MODELS_TO_TEST, CUSTOM_QUERY_PROMPT, GROUND_TRUTH_DATA
from chroma_db_utils import get_chroma_path, get_chroma_collection_and_vector_store


llama_debug_handler = LlamaDebugHandler(print_trace_on_end=True)
Settings.callback_manager = CallbackManager([llama_debug_handler])


def evaluate_rag_models(results_base_dir="evaluation_results"):
    print(f"--- Uruchamianie ewaluacji RAG dla modeli LLM (LlamaIndex Evaluation) ---")

    
    results_base_dir = results_base_dir
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_base_dir, timestamp)  

    os.makedirs(results_dir, exist_ok=True)  
    print(f"Wyniki będą zapisywane do katalogu: {results_dir}")

    
    os.makedirs(CHROMA_PATH, exist_ok=True)
    db = chromadb.PersistentClient(path=CHROMA_PATH)

    
    all_evaluation_results = []

    
    EVALUATOR_LLM = Ollama(model="llama3.1:8b", request_timeout=300.0) 
    print(f"Skonfigurowano LLM 'Sędzia' dla ewaluatorów: {EVALUATOR_LLM.model}")
   

    
    faithfulness_evaluator = evaluation.FaithfulnessEvaluator(llm=EVALUATOR_LLM)
    answer_relevancy_evaluator = evaluation.AnswerRelevancyEvaluator(llm=EVALUATOR_LLM)
    context_relevancy_evaluator = evaluation.ContextRelevancyEvaluator(llm=EVALUATOR_LLM)
    correctness_evaluator = CorrectnessEvaluator(llm=EVALUATOR_LLM)

    
    detailed_results_json_path = os.path.join(results_dir, "evaluation_detailed_results_llama_index.json")

    
    if os.path.exists(detailed_results_json_path):
        os.remove(detailed_results_json_path)
        print(f"Usunięto istniejący plik szczegółowych wyników JSON: {detailed_results_json_path}")

   
    detailed_json_file = open(detailed_results_json_path, 'a', encoding='utf-8')

    for llm_model_name in LLM_MODELS_TO_TEST:
        print(f"\n==================== Ewaluacja dla modelu LLM: {llm_model_name} ====================")

        llm = Ollama(model=llm_model_name, request_timeout=300.0)
        Settings.llm = llm  
        print(f"Skonfigurowano LLM (testowany): {llm_model_name}")

       
        safe_llm_name = llm_model_name.replace(':', '_').replace('/', '_')
        llm_answers_filepath = os.path.join(results_dir, f"generated_answers_{safe_llm_name}.txt")
        if os.path.exists(llm_answers_filepath):
            with open(llm_answers_filepath, 'a', encoding='utf-8') as f:
                f.write("\n\n" + "=" * 80 + "\nNowe uruchomienie\n" + "=" * 80 + "\n\n")

        current_llm_answers_file = open(llm_answers_filepath, 'a', encoding='utf-8')

        
        for config in COLLECTIONS_CONFIG:
            collection_name = config["name"]
            embedding_model_name = config["embedding_model"]

            embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
            Settings.embed_model = embed_model  
            print(f"Skonfigurowano model embeddingowy: {embedding_model_name} dla kolekcji {collection_name}")

        
            chroma_collection, vector_store = get_chroma_collection_and_vector_store(db, collection_name)

            if chroma_collection.count() == 0:
                print(f"Kolekcja '{collection_name}' jest pusta. Pomiń tworzenie indeksu i silnika zapytań.")
                current_llm_answers_file.write(f"\n------ Kolekcja: {collection_name} ({embedding_model_name}) ------\n")
                current_llm_answers_file.write("Kolekcja jest pusta. Pomijam pytania.\n")
                continue

            collection_count = chroma_collection.count()
            dynamic_top_k = min(10, max(1, int(0.20 * collection_count)))
            print(f"  Dynamiczne similarity_top_k dla kolekcji {collection_name}: {dynamic_top_k}")

            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            retriever = VectorIndexRetriever(index=index, similarity_top_k=dynamic_top_k)
            response_synthesizer = get_response_synthesizer(response_mode="compact")
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[],
            )

            
            current_llm_answers_file.write(f"\n------ Kolekcja: {collection_name} ({embedding_model_name}) ------\n")

            for q_idx, question in enumerate(STATIC_QUESTIONS):
                print(f"\n--- Pytanie: {question} ---")

                evaluation_data = {
                    "question": question,
                    "llm_model_name": llm_model_name,
                    "embedding_model_name": embedding_model_name,
                    "embedding_collection": collection_name,
                    "generated_answer": "",
                    "retrieved_contexts": [],
                    "ground_truth_answer": None,  
                    "ground_truth_contexts": None,  
                    "time_elapsed_seconds": 0.0,
                    "generated_answer_word_count": None,
                    "faithfulness_score": None,
                    "answer_relevancy_score": None,
                    "context_relevancy_score": None,
                    "correctness_score": None,
                    "error": None
                }

                ground_truth_entry = GROUND_TRUTH_DATA.get(question)
                if not ground_truth_entry:
                    print(f"Brak danych referencyjnych dla pytania: '{question}'. Pomijam ewaluację.")
                    current_llm_answers_file.write(
                        f"Pytanie {q_idx + 1}: {question}\nOdpowiedź: Brak danych referencyjnych. Pominięto.\n\n"
                    )
                    evaluation_data["error"] = "Brak danych referencyjnych. Pominięto ewaluację."
                    all_evaluation_results.append(evaluation_data)  
                    continue

                ground_truth_answer = ground_truth_entry["answer"]
                ground_truth_contexts = ground_truth_entry["contexts"]

                
                evaluation_data["ground_truth_answer"] = ground_truth_answer
                evaluation_data["ground_truth_contexts"] = ground_truth_contexts

                print(f"  > Używam kolekcji: {collection_name}")
                start_time = time.time()

                generated_answer = ""
                retrieved_contexts = []
                time_elapsed = 0.0
                generated_answer_word_count = 0

                try:
                    response = query_engine.query(CUSTOM_QUERY_PROMPT.format(query_str=question))
                    retrieved_contexts = [n.node.get_content() for n in response.source_nodes]
                    generated_answer = str(response)
                    generated_answer_word_count = len(generated_answer.split())

                    time_elapsed = time.time() - start_time

                    print(f"  Wygenerowana Odpowiedź: '{generated_answer}'")
                    print(f"  Odzyskane Contexts (len): {len(retrieved_contexts)}")
                    print(f"  Czas odpowiedzi: {time_elapsed:.2f} s")
                    print(f"  Liczba słów w odpowiedzi: {generated_answer_word_count}")

                    
                    try:
                        faithfulness_result = faithfulness_evaluator.evaluate_response(
                            query=question, response=response
                        )
                        evaluation_data["faithfulness_score"] = faithfulness_result.score
                        print(f"  Faithfulness Score: {faithfulness_result.score} (Feedback: {faithfulness_result.feedback})")
                    except Exception as eval_e:
                        print(f"  Błąd podczas ewaluacji Faithfulness: {eval_e}")
                        evaluation_data["faithfulness_score"] = None

                    try:
                        answer_relevancy_result = answer_relevancy_evaluator.evaluate_response(
                            query=question, response=response
                        )
                        evaluation_data["answer_relevancy_score"] = answer_relevancy_result.score
                        print(f"  Answer Relevancy Score: {answer_relevancy_result.score} (Feedback: {answer_relevancy_result.feedback})")
                    except Exception as eval_e:
                        print(f"  Błąd podczas ewaluacji Answer Relevancy: {eval_e}")
                        evaluation_data["answer_relevancy_score"] = None

                    try:
                        context_relevancy_result = context_relevancy_evaluator.evaluate(
                            query=question,
                            contexts=retrieved_contexts
                        )
                        evaluation_data["context_relevancy_score"] = context_relevancy_result.score
                        print(f"  Context Relevancy Score: {context_relevancy_result.score} (Feedback: {context_relevancy_result.feedback})")
                    except Exception as eval_e:
                        print(f"  Błąd podczas ewaluacji Context Relevancy: {eval_e}")
                        evaluation_data["context_relevancy_score"] = None

                    try:
                        correctness_result = correctness_evaluator.evaluate_response(
                            query=question,
                            response=response,
                            reference_answer=ground_truth_answer
                        )
                        evaluation_data["correctness_score"] = correctness_result.score
                        print(f"  Correctness Score: {correctness_result.score} (Feedback: {correctness_result.feedback})")
                    except Exception as eval_e:
                        print(f"  Błąd podczas ewaluacji Correctness: {eval_e}")
                        evaluation_data["correctness_score"] = None

                    evaluation_data.update({
                        "generated_answer": generated_answer,
                        "retrieved_contexts": retrieved_contexts,
                        "time_elapsed_seconds": time_elapsed,
                        "generated_answer_word_count": generated_answer_word_count,
                    })

                except Exception as e:  
                    print(f"  Błąd podczas przetwarzania pytania dla kolekcji {collection_name}: {e}")
                    evaluation_data["error"] = str(e)

                
                all_evaluation_results.append(evaluation_data)

                
                answer_block = f"Pytanie {q_idx + 1}: {question}\n"
                answer_block += f"Odpowiedź: {generated_answer}\n"
                answer_block += f"Czas: {time_elapsed:.2f} s | Liczba słów: {generated_answer_word_count}\n"

                
                answer_block += "Metryki:\n"
                answer_block += (
                    f"  Faithfulness: {evaluation_data['faithfulness_score']:.2f} "
                    if evaluation_data['faithfulness_score'] is not None else "  Faithfulness: N/A "
                )
                answer_block += (
                    f"  Answer Relevancy: {evaluation_data['answer_relevancy_score']:.2f} "
                    if evaluation_data['answer_relevancy_score'] is not None else "  Answer Relevancy: N/A "
                )
                answer_block += (
                    f"  Context Relevancy: {evaluation_data['context_relevancy_score']:.2f} "
                    if evaluation_data['context_relevancy_score'] is not None else "  Context Relevancy: N/A "
                )
                answer_block += (
                    f"  Correctness: {evaluation_data['correctness_score']:.2f}\n"
                    if evaluation_data['correctness_score'] is not None else "  Correctness: N/A\n"
                )

                if evaluation_data["error"]:
                    answer_block += f"Błąd: {evaluation_data['error']}\n"
                answer_block += "\n"
                current_llm_answers_file.write(answer_block)
                current_llm_answers_file.flush()  

                detailed_json_file.write(json.dumps(evaluation_data, ensure_ascii=False) + '\n')
                detailed_json_file.flush()  

        
        current_llm_answers_file.close()  
        print(f"\nWygenerowane odpowiedzi dla {llm_model_name} zapisano do: {llm_answers_filepath}")

        
        summary_table_path = os.path.join(results_dir, "evaluation_summary_table_llama_index.csv")

        df_results_for_summary = pd.DataFrame(all_evaluation_results)

        summary_df = df_results_for_summary.groupby(['llm_model_name', 'embedding_model_name']).agg(
            avg_faithfulness=('faithfulness_score', 'mean'),
            avg_answer_relevancy=('answer_relevancy_score', 'mean'),
            avg_context_relevancy=('context_relevancy_score', 'mean'),
            avg_correctness=('correctness_score', 'mean'),
            avg_time_elapsed=('time_elapsed_seconds', 'mean'),
            avg_generated_words=('generated_answer_word_count', 'mean'),
            num_evaluated_questions=('question', 'count')
        ).reset_index()

        summary_df.to_csv(summary_table_path, index=False, encoding='utf-8')

        print(f"Tabela porównawcza ewaluacji zapisana do: {summary_table_path} (CSV)")

    
    detailed_json_file.close()
    print("\nZakończono ewaluację wszystkich modeli LLM.")


if __name__ == "__main__":
    evaluate_rag_models()