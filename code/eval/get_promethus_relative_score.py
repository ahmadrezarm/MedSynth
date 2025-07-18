
from utils.prometheus import get_and_save_prometheus_preference_scores, load_relative_judge, Note_2_Dial_get_and_save_prometheus_preference_scores
from utils import constants


# edit this for every model you wanna run.

comparisons_to_run= {"pair1": {"model_A_name": "Aci_Train_Only",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/aci_train_model_2024-07-19.csv",
                               "model_B_name": "Base_model_No_Tune",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_base_model_no_tuning_2025-02-08.csv"},
                               

                     "pair2": {"model_A_name": "MS_10k_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/AhmadData_10k_samples_and_aci_train_model_2024-08-11.csv",
                               "model_B_name": "Base_model_No_Tune",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_base_model_no_tuning_2025-02-08.csv"},
                               
                     "pair3": {"model_A_name": "MS_10k_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/AhmadData_10k_samples_and_aci_train_model_2024-08-11.csv",
                               "model_B_name": "NC_10k_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/NoteChat_10k_samples_and_aci_train_model_2024-08-11.csv"},

                        "pair4": {"model_A_name": "MS_10k_Only",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Ahmad_10k_samples_Only_model_2024-08-17.csv",
                               "model_B_name": "NC_10k_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/NoteChat_10k_samples_Only_model_2024-08-16.csv"},

                        "pair5": {"model_A_name": "MS_57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Ahmad57_and_Aci_2024-12-14.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Primock57_and_Aci_2024-12-14.csv"},

                        "pair6": {"model_A_name": "MS_57_Only",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Ahmad57_Only_2024-12-14.csv",
                               "model_B_name": "PriMock_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Primock57_Only_2024-12-14.csv"},

                        "pair7": {"model_A_name": "Ab_NoJudge_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NoJudge_and_Aci_2025-02-08.csv",
                               "model_B_name": "AllGPT_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_Ahmad_246_Aci_2025-02-08.csv"},

                  }



comparisons_to_run_v2 = {"pair1": {"model_A_name": "Ab_Ab_NoJudge_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NoJudge_and_Aci_2025-02-08.csv",
                               "model_B_name": "LammaJudgeGPT_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_LammaJudgeGPT_and_Aci_2025-02-08.csv"},
        
                            "pair2": {"model_A_name": "Ab_AllLamma57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_AllLamma_57Sample_and_Aci_2025-02-08.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Primock57_and_Aci_2024-12-14.csv"},

                            "pair3": {"model_A_name": "Ab_AllQwen57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/AllQwen_57_Samples_and_Aci_2025-02-08.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Primock57_and_Aci_2024-12-14.csv"},

                            "pair4": {"model_A_name": "Ab_AllLamma_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_AllLamma_and_Aci_2025-02-08.csv",
                               "model_B_name": "NC_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NC_246_Samples_and_Aci_2025-02-08.csv"},

                            "pair5": {"model_A_name": "Ab_AllQwen_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_AllQwen_and_Aci_2025-02-08.csv",
                               "model_B_name": "NC_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NC_246_Samples_and_Aci_2025-02-08.csv"},

                            }




N2D_comparisons_to_run = {"pair1": {"model_A_name": "N2D_Aci_Train_Only",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_Aci-Train_Only_2024-08-30.csv",
                               "model_B_name": "No_Tuning",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_base_model_no_tuning_2025-02-08.csv"},

                           "pair2": {"model_A_name": "N2D_MS_10k_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_Ahmad_and_Aci_2024-08-30.csv",
                               "model_B_name": "NC_10k_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_NoteChat_and_Aci_2024-08-30.csv"},

                           "pair3": {"model_A_name": "N2D_MS_10k_Only",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_Ahmad_Only_2024-08-30.csv",
                               "model_B_name": "NC_10k_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_NoteChat_Only_2024-08-30.csv"},


                           "pair4": {"model_A_name": "N2D_MS_57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Ahmad57_and_Aci_2024-12-14.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock57_and_Aci_2024-12-14.csv"},


                           "pair5": {"model_A_name": "N2D_MS_57_Only",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Ahmad57_Only_2024-12-14.csv",
                               "model_B_name": "PriMock_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock_Only_2024-12-09.csv"},


                           "pair6": {"model_A_name": "N2D_NoJudge_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_NoJudge_and_Aci_2025-02-08.csv",
                               "model_B_name": "AllGPT_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_Ahmad_246_Aci_2025-02-09.csv"},


                           "pair7": {"model_A_name": "N2D_NoJudge_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_NoJudge_and_Aci_2025-02-08.csv",
                               "model_B_name": "LammJudgeGPT_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_LammaJudgeGPT_and_Aci_2025-02-08.csv"},


                           "pair8": {"model_A_name": "N2D_AllLamma_57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_AllLamma_57Sample_and_Aci_2025-02-09.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock57_and_Aci_2024-12-14.csv"},


                           "pair9": {"model_A_name": "N2D_AllQwen_57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_AllQwen_57_Samples_and_Aci_2025-02-09.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock57_and_Aci_2024-12-14.csv"},


                           "pair10": {"model_A_name": "N2D_AllLamma_246_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_AllLamma_and_Aci_2025-02-08.csv",
                               "model_B_name": "NC_246_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_NC_246_Samples_and_Aci_2025-02-09.csv"},


                           "pair11": {"model_A_name": "N2D_AllQwen_246_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_AllQwen_and_Aci_2025-02-09.csv",
                               "model_B_name": "NC_246_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_NC_246_Samples_and_Aci_2025-02-09.csv"},
                        }



N2D_comparisons_to_run_remained = {"pair1": {"model_A_name": "N2D_MS_10k_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_Ahmad_and_Aci_2024-08-30.csv",
                               "model_B_name": "Aci_Train_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_Aci-Train_Only_2024-08-30.csv"},
                                }


comparisons_to_run_remained= {
                     "pair1": {"model_A_name": "MS_10k_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/AhmadData_10k_samples_and_aci_train_model_2024-08-11.csv",
                               "model_B_name": "Aci_train_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/aci_train_model_2024-07-19.csv"},
}

def main():
        relative_judge_model= load_relative_judge()
        for pair_name, pair_data in comparisons_to_run_remained.items():
                print(f"Pair is {pair_name} and Model A name is: {pair_data['model_A_name']}")
                get_and_save_prometheus_preference_scores(model_A_name= pair_data["model_A_name"],
                                                        model_B_name= pair_data["model_B_name"],
                                                        model_A_dial_summary_pairs_path= pair_data["model_A_data_path"],
                                                        model_B_dial_summary_pairs_path= pair_data["model_B_data_path"],
                                                        test_data_path=constants.Aci_test_path,
                                                        path_to_save= "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/llm_eval_v2/reltive_scores",
                                                        relative_judge_model= relative_judge_model)



if __name__ == '__main__':
        main()