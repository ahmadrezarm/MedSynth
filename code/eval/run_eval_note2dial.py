from utils import note2dial_model_evaluator
from utils import constants

# change this whenever you have a new model.
def main():
    evaluator= note2dial_model_evaluator.ModelEvaluatorAutoMetrics("Ahmad0067/SynthDataGen_Note2Dial_llama-3-8b-Instruct_Ahmad_Only")  
    dial_summary_pairs = evaluator.get_model_responses()
    evaluator.save_model_output_to_csv(dial_summary_pairs = dial_summary_pairs, name= "Note2Dial_llama-3-8b-Instruct_Ahmad_Only")
    eval_metrics= evaluator.get_automatic_eval_scores(dial_summary_pairs= dial_summary_pairs)
    evaluator.save_eval_metrics_to_csv(eval_metrics= eval_metrics, metrics_filename= "Note2Dial_llama-3-8b-Instruct_Ahmad_Only")


if __name__ == '__main__':
    main()
