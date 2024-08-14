import model_evaluator
import constants

# change this whenever you have a new model.
def main():
    evaluator = model_evaluator.ModelEvaluatorAutoMetrics(
        model=constants.gpt_config["model"],
    )
    dial_summary_pairs = evaluator.get_gpt_responses()
    evaluator.save_model_output_to_csv(
        dial_summary_pairs=dial_summary_pairs, 
        model_name= constants.gpt_config["model"]
    )
    
    # Evaluate the GPT responses
    eval_metrics = evaluator.get_automatic_eval_scores(
        dial_summary_pairs=dial_summary_pairs,
        model_name=constants.gpt_config["model"]
    )
    
    # Save the evaluation metrics to the cumulative metrics CSV file
    evaluator.save_eval_metrics_to_csv()

if __name__ == '__main__':
    main()