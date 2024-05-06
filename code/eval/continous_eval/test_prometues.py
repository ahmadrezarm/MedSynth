import pandas as pd

from prometheus2 import get_preference_score, get_absolute_score
from utils import constants

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