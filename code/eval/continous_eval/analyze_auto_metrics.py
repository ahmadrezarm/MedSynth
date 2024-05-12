import pandas as pd
import numpy as np

df_base= pd.read_csv("/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/eval_results/cont_eval/auto_metics_base_model_2024-05-12.csv", sep="|")
df_aci= pd.read_csv("/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/eval_results/cont_eval/auto_metics_aci_train_model_2024-05-12.csv", sep="|")

for col in df_base.columns:
    print(f"Base Model {col} Score is: {df_base[col]}")
    print(f"Aci Train Model {col} Score is: {df_aci[col]}")

