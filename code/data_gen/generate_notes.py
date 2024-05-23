from gen_utils import note_generation_functions
from gen_utils import gen_constants



# edit this whenver you need a different medical condition or different count
def main():
    openai_client= note_generation_functions.initialize_openai_client()
    ### to add versions to the name, go to the note_generation_functions.py
    note_generation_functions.generate_and_save_medical_notes(disease_description= "CHRONIC PAIN SYNDROME", 
                                                              notes_count= 8, 
                                                              openai_client= openai_client,
                                                              path_to_save_notes= gen_constants.PATH_TO_SAVE_NOTES)



if __name__ == '__main__':
    main()