import pandas as pd
import numpy as np


df_aci= pd.read_csv("/Users/ahmadrezaie/2_My_papers/Synthetic_Data_Gen/data/eval_results/cont_eval/auto_metics_aci_train_model_2024-07-19.csv", sep="|")
df_noteChat= pd.read_csv("/Users/ahmadrezaie/2_My_papers/Synthetic_Data_Gen/data/eval_results/cont_eval/auto_metics_NoteChat_4k_samples_and_aci_train_model_2024-08-08.csv", sep="|")
df_ahmad_aci= pd.read_csv("/Users/ahmadrezaie/2_My_papers/Synthetic_Data_Gen/data/eval_results/cont_eval/auto_metics_AhmadData_4k_samples_and_aci_train_model_2024-08-08.csv", sep="|")

for col in df_aci.columns:
    print(f"Aci Train  Model {col} Score is: {df_aci[col]}")
    print(f"NoteChat 4k Sample Aci Train  Model {col} Score is: {df_noteChat[col]}")
    print(f"Ahmad 4k Samples Aci Train Model {col} Score is: {df_ahmad_aci[col]}")
    print("*******************")

