
from utils.prometheus import get_and_save_prometheus_absolute_scores, load_absolute_judge

# edit this for every model you wanna run.
#                         "aci_train_model": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/aci_train_model_2024-07-19.csv",
models_to_get_scores= { 
                        "Aci_Aci_and_Ahmad_56": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Aci_Aci_and_Ahmad_56_2024-12-08.csv",
                       "ACI_Primock57_and_Aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Primock57_and_Aci_2024-12-14.csv",
                       "Aci_Ahmad_56_only": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Aci_Ahmad_56_only_2024-12-08.csv",
                       "ACI_Primock57_Only": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Primock57_Only_2024-12-14.csv",
                       "ACI_AhmadData_5.5k_samples_and_aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/AhmadData_5.5k_samples_and_aci_train_model_2024-08-09.csv",
                       "ACI_NoteChat_5.5k_samples_and_aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/NoteChat_5.5k_samples_and_aci_train_model_2024-08-09.csv",
                       "ACI_NoteChat_10k_samples_Only": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/NoteChat_10k_samples_Only_model_2024-08-16.csv",
                       "Ahmad_10k_samples_Only": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Ahmad_10k_samples_Only_model_2024-08-17.csv",
                       "NoteChat_10k_samples_and_aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/NoteChat_10k_samples_and_aci_train_model_2024-08-11.csv",
                       }


models_to_get_scores2= { "ACI_Ahmad_246_Aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_Ahmad_246_Aci_2025-02-08.csv",
                        "ACI_AllLamma_57Sample_and_Aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_AllLamma_57Sample_and_Aci_2025-02-08.csv",
                        "ACI_AllLamma_and_Aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_AllLamma_and_Aci_2025-02-08.csv",
                        "ACI_AllQwen_and_Aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_AllQwen_and_Aci_2025-02-08.csv",
                        "ACI_base_model_no_tuning": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_base_model_no_tuning_2025-02-08.csv",
                        "ACI_LammaJudgeGPT_and_Aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_LammaJudgeGPT_and_Aci_2025-02-08.csv",
                        "ACI_NC_246_Samples_and_Aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NC_246_Samples_and_Aci_2025-02-08.csv",      
                        "ACI_NoJudge_and_Aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NoJudge_and_Aci_2025-02-08.csv",
                        "AllQwen_57_Samples_and_Aci": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/AllQwen_57_Samples_and_Aci_2025-02-08.csv",                   
}


models_to_get_scores_corections = {
                        "Aci_Aci_and_Ahmad_57": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Ahmad57_and_Aci_2024-12-14.csv",
                       "Aci_Ahmad_57_only": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Ahmad57_Only_2024-12-14.csv",




}

def main():
        absolute_judge= load_absolute_judge()
        for model_name, data_path in models_to_get_scores_corections.items():
                print(f"working on: {model_name}")
                get_and_save_prometheus_absolute_scores(model_name= model_name,
                                                        dial_summary_pairs_path= data_path, 
                                                        save_path= "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/llm_eval_v2",
                                                        absolute_judge= absolute_judge)





if __name__ == '__main__':
        main()