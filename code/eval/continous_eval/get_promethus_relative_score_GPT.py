
import os
import litellm

from utils.prometheus import get_and_save_prometheus_preference_scores_GPT, Note2Dial_get_and_save_prometheus_preference_scores_GPT
from utils import constants


litellm.set_verbose=True
os.environ['LITELLM_LOG'] = 'DEBUG'
os.environ["OPENAI_API_KEY"] = "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"
# edit this for every model you wanna run.
""" 
def main():
        get_and_save_prometheus_preference_scores_GPT(model_A_name= "ACI_Ahmad56_and_Aci",
                                                  model_B_name= "Primock_and_Aci",
                                                  model_A_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/Aci_Aci_and_Ahmad_56_2024-12-08.csv",
                                                  model_B_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/Aci_Aci_train_and_Primock_2024-12-08.csv",
                                                  test_data_path=constants.Aci_test_path) 

""" 

def main():
        Note2Dial_get_and_save_prometheus_preference_scores_GPT(model_A_name= "ACI_Ahmad56_and_Aci",
                                                  model_B_name= "Primock_and_Aci",
                                                  model_A_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Ahmad56_and_Aci_2024-12-09.csv",
                                                  model_B_dial_summary_pairs_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock_and_Aci_2024-12-09.csv",
                                                  test_data_path= constants.Aci_test_path)  #


if __name__ == '__main__':
        main()