import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

base_model_results= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_base_model_2024-05-12.csv", sep="|")
aci_train_model_results= pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_aci_train_v2_2024-05-13.csv",sep= "|")


# Base model calculations
print(f"Base model score mean is: {np.mean(base_model_results['Score'])}")
print(f"Base model score std is: {np.std(base_model_results['Score'], ddof=1)}")  # Using Bessel's correction

# ACI train model calculations
print(f"Aci train model score mean is: {np.mean(aci_train_model_results['Score'])}")
print(f"Aci train model score std is: {np.std(aci_train_model_results['Score'], ddof=1)}")  # Using Bessel's correction

base_score_counts = base_model_results['Score'].value_counts()
aci_score_counts = aci_train_model_results['Score'].value_counts()


print(f"Base Model: {base_score_counts}")
print(f" Aci Model v2: {aci_score_counts}")


# Filter the rows where Score is equal to 1
filtered_data = aci_train_model_results[aci_train_model_results['Score'] == 1]

# Loop over each row in the filtered DataFrame
for index, row in filtered_data.iterrows():
    print(f"Note is: {row['model_note']}")
    print(f"Feedback is: {row['feedback']}")

