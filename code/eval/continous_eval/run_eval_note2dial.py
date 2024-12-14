from utils import note2dial_model_evaluator
from utils import constants

""" 
# change this whenever you have a new model.
def main():
    evaluator= note2dial_model_evaluator.ModelEvaluatorAutoMetrics("Ahmad0067/SynthDataGen_Note2Dial_llama-3-8b-Instruct_ACI_Primock_Only")  
    dial_summary_pairs = evaluator.get_model_responses()
    evaluator.save_model_output_to_csv(dial_summary_pairs = dial_summary_pairs, name= "N2D_ACI_Primock_Only")
    eval_metrics= evaluator.get_automatic_eval_scores(dial_summary_pairs= dial_summary_pairs)
    evaluator.save_eval_metrics_to_csv(eval_metrics= eval_metrics, metrics_filename= "N2D_ACI_Primock_Only")


if __name__ == '__main__':
    main()
""" 


N2D_eval_models_dict= {"N2D_ACI_Ahmad56_and_Aci": "Ahmad0067/SynthDataGen_Note2Dial_llama-3-8b-Instruct_Ahmad_56_and_Aci",
                   "N2D_ACI_Ahmad56_Only": "Ahmad0067/SynthDataGen_Note2Dial_llama-3-8b-Instruct_Ahmad_56_Only",
                   "N2D_ACI_Primock_and_Aci": "Ahmad0067/SynthDataGen_Note2Dial_llama-3-8b-Instruct_ACI_Primock_and_Aci",
                   }
#                   "N2D_ACI_Primock_Only": "Ahmad0067/SynthDataGen_Note2Dial_llama-3-8b-Instruct_ACI_Primock_Only"

# change this whenever you have a new model.
def main():
    for model, hf_path in N2D_eval_models_dict.items():
        print(model)
        print(hf_path)
        print(20*"*")
        evaluator= note2dial_model_evaluator.ModelEvaluatorAutoMetrics(hf_path)  #"Ahmad0067/SynthDataGen_Dial2Note_llama-3-8b-Instruct_Ahmad_56_and_Aci"
        dial_summary_pairs = evaluator.get_model_responses()
        evaluator.save_model_output_to_csv(dial_summary_pairs = dial_summary_pairs, name= model) #"Aci_Aci_and_Ahmad_56"
        eval_metrics= evaluator.get_automatic_eval_scores(dial_summary_pairs= dial_summary_pairs)
        evaluator.save_eval_metrics_to_csv(eval_metrics= eval_metrics, metrics_filename= model) #"Aci_Aci_and_Ahmad_56"


if __name__ == '__main__':
    main()

