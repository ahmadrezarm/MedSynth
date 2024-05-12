import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

base_model_results= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_base_model_2024-05-12.csv", sep="|")
aci_train_model_results= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_aci_train_model_2024-05-12.csv",sep= "|")

# Exclude row 10 from calculations
#base_model_except_10 = base_model_results.drop(10)
#aci_train_except_10 = aci_train_model_results.drop(10)

# Base model calculations
print(f"Base model score mean is: {np.mean(base_model_results['Score'])}")
print(f"Base model score std is: {np.std(base_model_results['Score'], ddof=1)}")  # Using Bessel's correction

# ACI train model calculations
print(f"Aci train model score mean is: {np.mean(aci_train_model_results['Score'])}")
print(f"Aci train model score std is: {np.std(aci_train_model_results['Score'], ddof=1)}")  # Using Bessel's correction

base_score_counts = base_model_results['Score'].value_counts()
aci_score_counts = aci_train_model_results['Score'].value_counts()


print(base_score_counts)
print(aci_score_counts)

