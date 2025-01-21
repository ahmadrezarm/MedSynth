from bench_utils import model_evaluator, constants

# change this whenever you have a new model.
def main():
    evaluator = model_evaluator.ModelEvaluatorAutoMetrics(
        model= "Ahmad0067/SynthDataGen_Benchmarking_mistral-7b-instruct-v0.3", #constants.gpt_config["model"]
    )
    dial_summary_pairs = evaluator.get_mistral_responses()
    evaluator.save_model_output_to_csv(
        dial_summary_pairs=dial_summary_pairs, 
        model_name= "mistral-7b-instruct-v0.3"        #constants.gpt_config["model"]
    )
    
    # Evaluate the  responses
    eval_metrics = evaluator.get_automatic_eval_scores(
        dial_summary_pairs=dial_summary_pairs,
        model_name=  "mistral-7b-instruct-v0.3"       #constants.gpt_config["model"]
    )
    
    # Save the evaluation metrics to the cumulative metrics CSV file
    evaluator.save_eval_metrics_to_csv()

if __name__ == '__main__':
    main()