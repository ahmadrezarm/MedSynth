import pandas as pd
import numpy as np


df_aci= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/auto_metics_aci_train_model_2024-07-19.csv", sep="|")
df_noteChat= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/auto_metics_NoteChat_10k_samples_Only_model_2024-08-16.csv", sep="|")
df_ahmad= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/auto_metics_Ahmad_10k_samples_Only_model_2024-08-17.csv", sep="|")

for col in df_aci.columns:
    print(f"Aci Train  Model {col} Score is: {df_aci[col]}")
    print(f"NoteChat 4k Sample Train  Model {col} Score is: {df_noteChat[col]}")
    print(f"Ahmad 4k Samples Train Model {col} Score is: {df_ahmad[col]}")
    print("*******************")

