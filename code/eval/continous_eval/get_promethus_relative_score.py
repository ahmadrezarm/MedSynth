
from utils.prometheus import get_and_save_prometheus_preference_scores

# edit this for every model you wanna run.

def main():
        get_and_save_prometheus_preference_scores(model_A_name= "base_model_v3",
                                                  model_B_name= "aci_train_model_v3",
                                                  model_A_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/base_model_2024-05-14.csv",
                                                  model_B_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/aci_train_v3_2024-05-14.csv")



if __name__ == '__main__':
        main()