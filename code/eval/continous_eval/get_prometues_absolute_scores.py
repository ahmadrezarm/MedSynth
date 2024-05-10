
from utils.prometheus import get_and_save_prometheus_absolute_scores

# edit this for every model you wanna run.

def main():
        get_and_save_prometheus_absolute_scores(model_name= "base_model",
                                       dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/base_model_2024-05-10.csv")



if __name__ == '__main__':
        main()