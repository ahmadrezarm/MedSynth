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
    openai_client= note_generation_functions.initialize_openai_client()
    disease_list= ["ESSENTIAL (PRIMARY) HYPERTENSION", "ENCOUNTER FOR GENERAL ADULT MEDICAL EXAMINATION WITHOUT ABNORMAL FINDINGS",
                   "ENCOUNTER FOR ROUTINE CHILD HEALTH EXAMINATION WITHOUT ABNORMAL FINDINGS",
                   "ENCOUNTER FOR IMMUNIZATION", "TYPE 2 DIABETES MELLITUS WITHOUT COMPLICATIONS",
                   "END STAGE RENAL DISEASE", " HYPERLIPIDEMIA, UNSPECIFIED", "LOW BACK PAIN",
                   "OBSTRUCTIVE SLEEP APNEA (ADULT) (PEDIATRIC)", " ILLNESS, UNSPECIFIED"]
    
    for disease in disease_list:
            note_generation_functions.generate_and_save_medical_notes(disease_description= disease, 
                                                              notes_count= 5, 
                                                              openai_client= openai_client,
                                                              path_to_save_notes= gen_constants.PATH_TO_SAVE_NOTES)

if __name__ == '__main__':
    main()