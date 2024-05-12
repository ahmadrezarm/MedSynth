import pandas as pd
import numpy as np
from scipy import stats

base_model_results= pd.read_csv("/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_base_model_2024-05-12.csv", sep="|")
aci_train_model_results= pd.read_csv("/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_aci_train_model_2024-05-12.csv",sep= "|")

# Exclude row 10 from calculations
base_model_except_10 = base_model_results.drop(10)
aci_train_except_10 = aci_train_model_results.drop(10)

# Base model calculations
print(f"Base model score mean is: {np.mean(base_model_except_10['Score'])}")
print(f"Base model score std is: {np.std(base_model_except_10['Score'], ddof=1)}")  # Using Bessel's correction (ddof=1)

# Assuming you need mode; uncomment below if needed
# print(f"Base model score mode is: {stats.mode(base_model_except_10['Score']).mode[0]}")

# ACI train model calculations
print(f"Aci train model score mean is: {np.mean(aci_train_except_10['Score'])}")
# Uncomment below if needed
# print(f"Aci train model score mode is: {stats.mode(aci_train_except_10['Score']).mode[0]}")
print(f"Aci train model idx 10 score is: {aci_train_model_results['Score'].iloc[10]}")  # Access using iloc for safety
print(f"Aci train model score std is: {np.std(aci_train_except_10['Score'], ddof=1)}")  # Using Bessel's correction