from utils import model_evaluator
from utils import constants

# change this whenever you have a new model.
def main():
    evaluator= model_evaluator.ModelEvaluatorAutoMetrics("Ahmad0067/SynthDataGen_Dial2Note_llama-3-8b-Instruct_AhmadData_10k_samples_and_Aci")  
    dial_summary_pairs = evaluator.get_model_responses()
    evaluator.save_model_output_to_csv(dial_summary_pairs = dial_summary_pairs, name= "AhmadData_10k_samples_and_aci_train_model")
    eval_metrics= evaluator.get_automatic_eval_scores(dial_summary_pairs= dial_summary_pairs)
    evaluator.save_eval_metrics_to_csv(eval_metrics= eval_metrics, metrics_filename= "AhmadData_10k_samples_and_aci_train_model")


if __name__ == '__main__':
    main()
