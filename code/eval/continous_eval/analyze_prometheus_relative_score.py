import pandas as pd
import numpy as np


df= pd.read_csv("/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_base_model_aci_train_model_2024-05-12.csv", sep="|")

print(df["Preference"].value_counts())

print(df["Preference"][10])