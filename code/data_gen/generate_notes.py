import pandas as pd
import os

from gen_utils import note_generation_functions
from gen_utils import gen_constants




def file_exists(disease_description, path_to_save_notes):
    """Check if a file for the disease description already exists in the directory."""
    for filename in os.listdir(path_to_save_notes):
        if disease_description in filename and filename.endswith('.csv'):
            return True
    return False

def main():
    ###### Edit this for each phase #############
    path_to_save_notes= "/Users/ahmadrezaie/2_My_papers/Synthetic_Data_Gen/data/output/notes_dial_pairs/Test_feedback"
    ##############################################
    counter= 1
    openai_client= note_generation_functions.initialize_openai_client()
    #df = pd.read_csv("/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/input/IQVIA/IQVIA_cleaned.csv",  sep="|")
    #top_100_icd10_desc = df['ICD10_desc'].head(100).tolist()
    #next_200_icd10_desc = df['ICD10_desc'].iloc[100:300].tolist()
    #next_50_icd10_desc = df['ICD10_desc'].iloc[300:350].tolist()
    #next_50_icd10_desc = df['ICD10_desc'].iloc[350:400].tolist()
    #next_100_icd10_desc = df['ICD10_desc'].iloc[400:500].tolist()
    #next_300_icd10_desc = df['ICD10_desc'].iloc[500:800].tolist()
    #next_300_icd10_desc = df['ICD10_desc'].iloc[800:1100].tolist()  
    #next_300_icd10_desc = df['ICD10_desc'].iloc[1100:1400].tolist()  
    #next_300_icd10_desc = df['ICD10_desc'].iloc[1400:1700].tolist()  
    #next_300_icd10_desc = df['ICD10_desc'].iloc[1700:2000].tolist()  
    #next_7_icd10_desc = df['ICD10_desc'].iloc[2000:2007].tolist()
    remained= ["URINARY TRACT INFECTION, SITE NOT SPECIFIED"]  


    for disease in remained:
            print(f"counter is: {counter}")
            print(disease)
            if not file_exists(disease, path_to_save_notes):
                note_generation_functions.generate_and_save_medical_notes(disease_description= disease, 
                                                              notes_count= 5, 
                                                              openai_client= openai_client,
                                                              path_to_save_notes= path_to_save_notes) 
            counter += 1

if __name__ == '__main__':
    main()






