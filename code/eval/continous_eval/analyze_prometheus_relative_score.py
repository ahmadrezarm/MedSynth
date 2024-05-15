import pandas as pd
import numpy as np


df= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_aci_train_model_v3_aci_note_chat_train_model_v3_2024-05-14.csv", sep="|")

print(df["Preference"].value_counts())
