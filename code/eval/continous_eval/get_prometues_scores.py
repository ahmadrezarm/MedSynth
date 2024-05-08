import pandas as pd
from datetime import datetime

from utils.prometheus import get_preference_score, get_absolute_score
from utils import constants

""" 
conversation_list = pd.read_csv(constants.Aci_test_path)["dialogue"]
refence_list = pd.read_csv(constants.Aci_test_path)["note_SOAP"]

model_note_list = pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/base_model.csv")["summary"]

def save_results_to_csv(scores, name, path = "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval"):
        df = pd.DataFrame.from_dict(scores, orient='index')
        
        full_path = f"{path}/{name}.csv"
        df.to_csv(full_path, index=False)



scores = get_absolute_score(conversations_list = conversation_list, 
                            reference_list= refence_list, 
                            model_response_list = model_note_list)

save_results_to_csv(scores= scores, name= "prometheus_absolute_scores_base_model_second_run")

"""

def _save_prometheus_scores(prometheus_scores, model_A_name, model_B_name= None, base_name= constants.PROMETHEUS_RESULT_BASE_NAME, 
                                path= constants.PATH_TO_SAVE_EVAL_OUTPUT):
                
                df = pd.DataFrame.from_dict(prometheus_scores, orient='index')
                current_date= datetime.now().strftime("%Y-%m-%d")

                if model_B_name:
                        full_name = f"{base_name}_{model_A_name}_{model_B_name}_{current_date}.csv"

                else: 
                       full_name = f"{base_name}_{model_A_name}_{current_date}.csv" 

                full_path = f"{path}/{full_name}.csv"
                df.to_csv(full_path, index=False)

    

def get_prometheus_absolute_scores(model_name,dial_summary_pairs):

        dial_summary_pairs_df = pd.DataFrame.from_dict(dial_summary_pairs, orient='index')

        prometheus_absolute_scores = get_absolute_score(conversations_list= dial_summary_pairs_df["conversation"], 
                            reference_list= pd.read_csv(constants.Aci_test_path)["note_SOAP"], 
                            model_response_list= dial_summary_pairs_df["summary"])
        
        _save_prometheus_scores(prometheus_scores= prometheus_absolute_scores, model_A_name= model_name)
        
        


def get_prometheus_preference_scores(model_A_name, model_B_name, 
                                     model_A_dial_summary_pairs, model_B_dial_summary_pairs):
        
        model_A_dial_summary_pairs_df = pd.DataFrame.from_dict(model_A_dial_summary_pairs, orient='index')
        model_B_dial_summary_pairs_df = pd.DataFrame.from_dict(model_B_dial_summary_pairs, orient='index')


        prometheus_preference_scores = get_preference_score(conversations_list= model_A_dial_summary_pairs_df["conversation"], 
                            reference_list= pd.read_csv(constants.Aci_test_path)["note_SOAP"], 
                            model_A_response_list= model_A_dial_summary_pairs_df["summary"],
                            model_B_response_list= model_B_dial_summary_pairs_df["summary"])
        
        _save_prometheus_scores(model_A_name= model_A_name, model_B_name= model_B_name, prometheus_scores= prometheus_preference_scores)