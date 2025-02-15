from utils import model_tuner
from utils import constants

tuning_model_dict= {
                   "ACI_Ahmad_246_Aci": "Ahmad0067/SynthDataGen_llama3_Dial2Note_Ahmad_246_Samples_and_Aci_instruct_2025-02-08_11-18",
                   "ACI_AllQwen_and_Aci": "Ahmad0067/SynthDataGen_llama3_Dial2Note_All_Qwen_and_Aci_instruct_2025-02-08_11-02",
                   "ACI_AllQwen_57_Samples_and_Aci": "Ahmad0067/SynthDataGen_llama3_Dial2Note_All_Qwen_57_Samples_and_Aci_instruct_2025-02-08_11-02",
                   }

tuning_model_dict_remained= {
                   "Aci_AllLamma_246_Only": "Ahmad0067/SynthDataGen_llama3_Dial2Note_AllLamma_246_Samples_Only_instruct_2025-02-09_20-52",
                   "Aci_AllQwen_246_Only": "Ahmad0067/SynthDataGen_llama3_Dial2Note_AllQwen_246_Samples_Only_instruct_2025-02-09_20-47",
                   }



"""
# change this whenever a new model is being trained on a new data
def main():
    tuner = model_tuner.ModelTuner(TRAINING_DATA_PATH_HF= "Ahmad0067/SynthDataGen_llama3_Note2Dial_ACI_Ahmad56_Only_instruct_dataset2024-12-09_08-19", 
                                  FINE_TUNED_MODEL_NAME= "SynthDataGen_Note2Dial_llama-3-8b-Instruct_Ahmad_56_Only", 
                                  tuning_config= constants.tuning_config, base_model= constants.base_model)
    
    tuner.model_train_and_save()
 """

def main():
    for model, hf_path in tuning_model_dict_remained.items():
        print(f"working on: {model}")
        print(f" The datset is: {hf_path}")
        print(20*"*")
        tuner = model_tuner.ModelTuner(TRAINING_DATA_PATH_HF= hf_path, 
                                  FINE_TUNED_MODEL_NAME= f"SynthDataGen_{model}", 
                                  tuning_config= constants.tuning_config, base_model= constants.base_model)
    
        tuner.model_train_and_save()
        print(f"Model: SynthDataGen_{model}  has been pushed to the hub successfully. Going to the next model ...")
        print(20*"#")

""" 
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
"""


if __name__ == '__main__':
    main()

