import pandas as pd
import numpy as np


df= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_aci_note-chat_model_v3_base_model_2024-06-07.csv", sep="|")

print(df["Preference"].value_counts())
