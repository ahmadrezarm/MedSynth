import pandas as pd

from prometheus2 import get_preference_score, get_absolute_score
from utils import constants

conversation_list = pd.read_csv(constants.Aci_test_path)["dialogue"]
refence_list = pd.read_csv(constants.Aci_test_path)["note_SOAP"]

model_note_list = pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/base_model.csv")["summary"]



