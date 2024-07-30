import pandas as pd

from gen_utils import note_generation_functions
from gen_utils import gen_constants



# edit this whenver you need a different medical condition or different count
'''
def main():
    openai_client= note_generation_functions.initialize_openai_client()
    ### to add versions to the name, go to the note_generation_functions.py
    note_generation_functions.generate_and_save_medical_notes(disease_description= "CHRONIC PAIN SYNDROME", 
                                                              notes_count= 8, 
                                                              openai_client= openai_client,
                                                              path_to_save_notes= gen_constants.PATH_TO_SAVE_NOTES)



'''

def main():
    counter= 1
    openai_client= note_generation_functions.initialize_openai_client()
    df = pd.read_csv("/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/input/IQVIA/IQVIA_cleaned.csv",  sep="|")
    #top_100_icd10_desc = df['ICD10_desc'].head(100).tolist()
    #next_200_icd10_desc = df['ICD10_desc'].iloc[100:300].tolist()
    #next_50_icd10_desc = df['ICD10_desc'].iloc[300:350].tolist()
    #next_50_icd10_desc = df['ICD10_desc'].iloc[350:400].tolist()
    #next_100_icd10_desc = df['ICD10_desc'].iloc[400:500].tolist()
    #next_300_icd10_desc = df['ICD10_desc'].iloc[500:800].tolist()
    next_300_icd10_desc = df['ICD10_desc'].iloc[800:1100].tolist()  ##### HAS NOT BEEN GENERATED YET######

    for disease in next_300_icd10_desc:
            print(f"counter is: {counter}")
            print(disease)
            note_generation_functions.generate_and_save_medical_notes(disease_description= disease, 
                                                              notes_count= 5, 
                                                              openai_client= openai_client,
                                                              path_to_save_notes= "/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/output/notes_onVector/phase_8_1.5k") #gen_constants.PATH_TO_SAVE_NOTES
            counter += 1
if __name__ == '__main__':
    main()







    ''' 
    disease_list= ['LONG TERM (CURRENT) USE OF AROMATASE INHIBITORS',
                    'OSTEOPHYTE, VERTEBRAE',
                    'PERSONAL HISTORY OF NON-HODGKIN LYMPHOMAS',
                    'WEDGE COMPRESSION FRACTURE OF FIRST LUMBAR VERTEBRA, INITIAL ENCOUNTER FOR CLOSED FRACTURE',
                    'CUTANEOUS ABSCESS OF GROIN']
    """
                        'ACQUIRED KERATOSIS [KERATODERMA] PALMARIS ET PLANTARIS',
                    'HEART DISEASE, UNSPECIFIED',
                    'SECONDARY AND UNSPECIFIED MALIGNANT NEOPLASM OF LYMPH NODE, UNSPECIFIED',
                    'GESTATIONAL DIABETES MELLITUS IN CHILDBIRTH, UNSPECIFIED CONTROL',
                    'INCOMPLETE ROTATOR CUFF TEAR OR RUPTURE OF RIGHT SHOULDER, NOT SPECIFIED AS TRAUMATIC',
                    'CERVICOBRACHIAL SYNDROME',
                    'GANGRENE, NOT ELSEWHERE CLASSIFIED',
                    'OTHER ABNORMAL FINDINGS ON DIAGNOSTIC IMAGING OF CENTRAL NERVOUS SYSTEM',
                    'DRY EYE SYNDROME OF BILATERAL LACRIMAL GLANDS',
                    'DISEASE OF INTESTINE, UNSPECIFIED'
    """
    
    
    [
                   "ENCOUNTER FOR GENERAL ADULT MEDICAL EXAMINATION WITHOUT ABNORMAL FINDINGS",
                   "ENCOUNTER FOR ROUTINE CHILD HEALTH EXAMINATION WITHOUT ABNORMAL FINDINGS",
                   "ENCOUNTER FOR IMMUNIZATION"]
    
   "ESSENTIAL (PRIMARY) HYPERTENSION", "TYPE 2 DIABETES MELLITUS WITHOUT COMPLICATIONS",
                   "END STAGE RENAL DISEASE", " HYPERLIPIDEMIA, UNSPECIFIED", "LOW BACK PAIN",
                   "OBSTRUCTIVE SLEEP APNEA (ADULT) (PEDIATRIC)", " ILLNESS, UNSPECIFIED"
    '''
