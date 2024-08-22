from bench_utils import model_evaluator, constants
import pandas as pd
evaluator = model_evaluator.ModelEvaluatorAutoMetrics(
 #constants.gpt_config["model"]
    )

dial_summary_pairs= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/benchmarking/results/llama-3-8b-Instruct_2024-08-18.csv", sep="|")
results_df= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/benchmarking/results/combined/benchmark_metrics.csv", sep="|")

eval_metrics = evaluator.get_automatic_eval_scores(
        dial_summary_pairs=dial_summary_pairs,
        model_name=  "llama-3-8b-Instruct"       
    )
eval_metrics_df = pd.DataFrame([eval_metrics])


results_df = pd.concat([results_df, eval_metrics_df], ignore_index=True)
results_df.to_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/benchmarking/results/combined/benchmark_metrics_v2.csv", index=False, sep="|")