from utils import model_evaluator
from utils import constants

""" 
# change this whenever you have a new model.
def main():
    evaluator= model_evaluator.ModelEvaluatorAutoMetrics("Ahmad0067/SynthDataGen_Dial2Note_llama-3-8b-Instruct_Ahmad_56_and_Aci")  
    dial_summary_pairs = evaluator.get_model_responses()
    evaluator.save_model_output_to_csv(dial_summary_pairs = dial_summary_pairs, name= "Aci_Aci_and_Ahmad_56")
    eval_metrics= evaluator.get_automatic_eval_scores(dial_summary_pairs= dial_summary_pairs)
    evaluator.save_eval_metrics_to_csv(eval_metrics= eval_metrics, metrics_filename= "Aci_Aci_and_Ahmad_56")
""" 
# "ACI_Ahmad57_and_Aci": "Ahmad0067/SynthDataGen_ACI_Ahmad57_and_Aci",
#                    "ACI_Ahmad57_Only": "Ahmad0067/SynthDataGen_ACI_Ahmad57_Only",
eval_models_dict= {
                   "ACI_base_model_no_tuning": constants.base_model,
                   "ACI_NoJudge_and_Aci": "Ahmad0067/SynthDataGen_ACI_NoJudge_and_Aci",
                   "ACI_LammaJudgeGPT_and_Aci": "Ahmad0067/SynthDataGen_ACI_LammaJudgeGPT_and_Aci",
                   "ACI_AllLamma_and_Aci": "Ahmad0067/SynthDataGen_ACI_AllLamma_and_Aci",
                   "ACI_AllLamma_57Sample_and_Aci": "Ahmad0067/SynthDataGen_ACI_AllLamma_57Sample_and_Aci",
                   "ACI_NC_246_Samples_and_Aci": "Ahmad0067/SynthDataGen_ACI_NC_246_Samples_and_Aci",
                   "ACI_Ahmad_246_Aci": "Ahmad0067/SynthDataGen_ACI_Ahmad_246_Aci",
                   "ACI_AllQwen_and_Aci": "Ahmad0067/SynthDataGen_ACI_AllQwen_and_Aci",
                   "AllQwen_57_Samples_and_Aci": "Ahmad0067/SynthDataGen_ACI_AllQwen_57_Samples_and_Aci",
                   }

eval_models_dict_remined= {
                   "Aci_AllLamma_246_Only":"Ahmad0067/SynthDataGen_Aci_AllLamma_246_Only",
                   "Aci_AllQwen_246_Only": "Ahmad0067/SynthDataGen_Aci_AllQwen_246_Only",
                   }

# change this whenever you have a new model.
def main():
    for model, hf_path in eval_models_dict_remined.items():
        print(f"working on: {model}")
        print(f" The model path is: {hf_path}")
        print(20*"*")
        evaluator= evaluator= model_evaluator.ModelEvaluatorAutoMetrics(hf_path) 
        dial_summary_pairs = evaluator.get_model_responses()
        evaluator.save_model_output_to_csv(dial_summary_pairs = dial_summary_pairs, name= model)
        eval_metrics= evaluator.get_automatic_eval_scores(dial_summary_pairs= dial_summary_pairs)
        evaluator.save_eval_metrics_to_csv(eval_metrics= eval_metrics, metrics_filename= model)



if __name__ == '__main__':
    main()
