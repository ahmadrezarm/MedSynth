import pandas as pd
import numpy as np

base_model_results= pd.read_csv("/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_base_model_2024-05-12.csv", sep="|")
aci_train_model_results= pd.read_csv("/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/eval_results/cont_eval/prometheus_scores_aci_train_model_2024-05-12.csv",sep= "|")

print(f"Base model score mean is: {np.mean(base_model_results['Score'])}")
print(f"Base model score std is: {np.std(base_model_results['Score'])}")

print(f"Aci train model score mean is: {np.mean(aci_train_model_results['Score'])}")
print(f"Aci train model score mean is: {np.std(aci_train_model_results['Score'])}")

