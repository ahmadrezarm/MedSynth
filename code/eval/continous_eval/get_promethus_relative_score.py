
from utils.prometheus import get_and_save_prometheus_preference_scores, Note_2_Dial_get_and_save_prometheus_preference_scores
from utils import constants


# edit this for every model you wanna run.
""" 
def main():
        get_and_save_prometheus_preference_scores(model_A_name= "ACI_Ahmad57_Only",
                                                  model_B_name= "Primock57_Only",
                                                  model_A_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Ahmad57_Only_2024-12-14.csv",
                                                  model_B_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Primock57_Only_2024-12-14.csv",
                                                  test_data_path=constants.Aci_test_path)  #"/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/benchmarking/dataset/test.csv")

"""


def main():
        Note_2_Dial_get_and_save_prometheus_preference_scores(model_A_name= "ACI_Ahmad57_Only",
                                                  model_B_name= "Primock57_Only",
                                                  model_A_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Ahmad57_Only_2024-12-14.csv",
                                                  model_B_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock_Only_2024-12-09.csv",
                                                  test_data_path= constants.Aci_test_path)  #



if __name__ == '__main__':
        main()