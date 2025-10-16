import os
import shutil
import time
import pandas as pd
import sys  
import datetime 
from evaluate_models import evaluate_rag_models


NUM_REPETITIONS = 10  
BASE_RESULTS_DIR = "evaluation_results_repetitive"  

def aggregate_results_from_runs(base_results_dir=BASE_RESULTS_DIR):
    """
    Zbiera pliki podsumowujące z każdego przebiegu i generuje jedno zbiorcze podsumowanie.
    """
    print(f"\n--- Rozpoczynanie agregacji wyników z katalogu: {base_results_dir} ---")

    all_summary_files = []
    for item in os.listdir(base_results_dir):
        run_path = os.path.join(base_results_dir, item)
        if os.path.isdir(run_path): 
            summary_file_path = os.path.join(run_path, "evaluation_summary_table_llama_index.csv")
            if os.path.exists(summary_file_path):
                all_summary_files.append(summary_file_path)

    if all_summary_files:
        print(f"Znaleziono {len(all_summary_files)} plików podsumowujących do agregacji.")
        try:
            
            list_of_dfs = [pd.read_csv(f) for f in all_summary_files]
            combined_df = pd.concat(list_of_dfs, ignore_index=True)

            
            all_possible_numeric_cols = [
                'avg_faithfulness', 'avg_answer_relevancy', 'avg_context_relevancy',
                'avg_correctness', 'avg_precision', 'avg_recall', 'avg_mrr',
                'avg_time_elapsed', 'avg_generated_words'
            ]

           
            actual_numeric_cols = [col for col in all_possible_numeric_cols if col in combined_df.columns]

            
            for col in actual_numeric_cols:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

            
            aggregation_dict = {}
            for col in actual_numeric_cols:
                aggregation_dict[f'{col}_mean'] = pd.NamedAgg(column=col, aggfunc='mean')
                aggregation_dict[f'{col}_median'] = pd.NamedAgg(column=col, aggfunc='median')
            
            
            if 'num_evaluated_questions' in combined_df.columns:
                aggregation_dict['num_evaluated_questions'] = pd.NamedAgg(column='num_evaluated_questions', aggfunc='sum')

            overall_summary_df = combined_df.groupby(['llm_model_name', 'embedding_model_name']).agg(
                **aggregation_dict
            ).reset_index()
            
            
            for col_key in aggregation_dict.keys(): 
                if col_key in overall_summary_df.columns and col_key != 'num_evaluated_questions':
                    overall_summary_df[col_key] = overall_summary_df[col_key].round(2)
            
            
            rename_map = {
                'avg_faithfulness_mean': 'avg_faithfulness',
                'avg_faithfulness_median': 'median_faithfulness',
                'avg_answer_relevancy_mean': 'avg_answer_relevancy',
                'avg_answer_relevancy_median': 'median_answer_relevancy',
                'avg_context_relevancy_mean': 'avg_context_relevancy',
                'avg_context_relevancy_median': 'median_context_relevancy',
                'avg_correctness_mean': 'avg_correctness',
                'avg_correctness_median': 'median_correctness',
                'avg_precision_mean': 'avg_precision',
                'avg_precision_median': 'median_precision',
                'avg_recall_mean': 'avg_recall',
                'avg_recall_median': 'median_recall',
                'avg_mrr_mean': 'avg_mrr',
                'avg_mrr_median': 'median_mrr',
                'avg_time_elapsed_mean': 'avg_time_elapsed',
                'avg_time_elapsed_median': 'median_time_elapsed',
                'avg_generated_words_mean': 'avg_generated_words',
                'avg_generated_words_median': 'median_generated_words',
            }
            filtered_rename_map = {old_name: new_name for old_name, new_name in rename_map.items() if old_name in overall_summary_df.columns}
            overall_summary_df = overall_summary_df.rename(columns=filtered_rename_map)

            overall_summary_path = os.path.join(base_results_dir, "evaluation_summary_table_overall.csv")
            overall_summary_df.to_csv(overall_summary_path, index=False)
            print(f"Zbiorcze podsumowanie zapisane do: {overall_summary_path}")
        except Exception as e:
            print(f"--- Błąd podczas generowania zbiorczego podsumowania: {e} ---")
    else:
        print(f"Brak plików podsumowujących CSV w katalogu {base_results_dir}/run_XX do agregacji. Żadne zbiorcze podsumowanie nie zostało wygenerowane.")

def run_repetitive_evaluation():
    print(f"--- Rozpoczynanie {NUM_REPETITIONS} powtórzeń ewaluacji ---")

    
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_repetitive")
    session_results_dir = os.path.join(BASE_RESULTS_DIR, session_timestamp)
    os.makedirs(session_results_dir, exist_ok=True)
    print(f"Wyniki powtarzalnych ewaluacji będą zapisywane do katalogu: {session_results_dir}")

    for i in range(NUM_REPETITIONS):
        
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_run")
        current_run_dir = os.path.join(session_results_dir, run_timestamp)
        
        print(f"\n--- Uruchomienie ewaluacji numer {i + 1}/{NUM_REPETITIONS} (Katalog: {current_run_dir}) ---")

        
        os.makedirs(current_run_dir, exist_ok=True)

        start_time_run = time.time()
        try:
            
            evaluate_rag_models(results_base_dir=current_run_dir)
            print(f"--- Ewaluacja numer {i + 1} zakończona pomyślnie. Czas: {time.time() - start_time_run:.2f} s ---")
        except Exception as e:
            print(f"--- Błąd podczas ewaluacji numer {i + 1}: {e} ---")
            print(f"--- Przebieg numer {i + 1} nie został zakończony pomyślnie. ---")
            continue 

    print(f"\n--- Zakończono wszystkie {NUM_REPETITIONS} powtórzenia ewaluacji. ---")
    print(f"Wszystkie wyniki indywidualne zostały zapisane w katalogu: {session_results_dir}")

    
    aggregate_results_from_runs(session_results_dir) 

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "aggregate":
        aggregate_results_from_runs(BASE_RESULTS_DIR)
    else:
        run_repetitive_evaluation()
