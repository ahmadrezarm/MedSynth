import pandas as pd
import numpy as np


df= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_Note_2_Dial_GPT_ACI_Ahmad56_and_Aci_Primock_and_Aci_2024-12-12.csv", sep="|")

print(df["Preference"].value_counts())
