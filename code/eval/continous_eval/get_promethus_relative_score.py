
from utils.prometheus import get_and_save_prometheus_preference_scores, Note_2_Dial_get_and_save_prometheus_preference_scores
from utils import constants


# edit this for every model you wanna run.
"""
def main():
        get_and_save_prometheus_preference_scores(model_A_name= "ACI_Ahmad56_Only",
                                                  model_B_name= "Primock_Only",
                                                  model_A_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/Aci_Ahmad_56_only_2024-12-08.csv",
                                                  model_B_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/Aci_Primock_Only_2024-12-08.csv",
                                                  test_data_path=constants.Aci_test_path) #"/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/benchmarking/dataset/test.csv")

""" 
# pairs_dict = {"experiment_1":{"Aci_Ahmad56": "",
#                               "Aci_PriMock": ""
#                               },
#               "experiment_2": {"Ahmad56_Only": "",
#                                "PriMock_Only": ""
#                                }
#                 }

def main():
        Note_2_Dial_get_and_save_prometheus_preference_scores(model_A_name= "ACI_Ahmad56_Only",
                                                  model_B_name= "Primock_Only",
                                                  model_A_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Ahmad56_Only_2024-12-09.csv",
                                                  model_B_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock_Only_2024-12-09.csv",
                                                  test_data_path= constants.Aci_test_path) #

 
if __name__ == '__main__':
        main()