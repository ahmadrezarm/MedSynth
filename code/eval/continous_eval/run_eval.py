from utils import model_evaluator
from utils import constants

# change this whenever you have a new model.
def main():
    evaluator= model_evaluator.ModelEvaluatorAutoMetrics(constants.base_model)   #"Ahmad0067/llama-3-8b-Instruct-bnb-4bit-aci-train_v3"
    dial_summary_pairs = evaluator.get_model_responses()
    evaluator.save_model_output_to_csv(dial_summary_pairs = dial_summary_pairs, name= "base_model")
    eval_metrics= evaluator.get_automatic_eval_scores(dial_summary_pairs= dial_summary_pairs)
    evaluator.save_eval_metrics_to_csv(eval_metrics= eval_metrics, metrics_filename= "base_model")


if __name__ == '__main__':
    main()