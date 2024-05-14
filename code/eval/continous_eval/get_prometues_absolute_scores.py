
from utils.prometheus import get_and_save_prometheus_absolute_scores

# edit this for every model you wanna run.

def main():
        get_and_save_prometheus_absolute_scores(model_name= "aci_train_v2",
                                                dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/aci_train_v2_2024-05-13.csv")



if __name__ == '__main__':
        main()