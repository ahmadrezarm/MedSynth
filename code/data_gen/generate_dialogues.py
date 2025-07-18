import os
import pandas as pd

from gen_utils import dial_generation_functions
#from .gen_utils import gen_constants


df = pd.read_csv("/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/input/IQVIA/IQVIA_cleaned.csv",  sep="|")
top_50_icd10_desc = df['ICD10_desc'].head(50).tolist()

### 1. change this
path_to_input= "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/output/notes_onVector/ablations_on_vector/all_qwen/"


def main():
    counter= 1
    openai_client= dial_generation_functions.initialize_openai_client()
    for filename in os.listdir(path_to_input):
        #print(filename)
        # only top 50 idcs
        if filename.endswith(".csv"):
            if any(desc in filename for desc in top_50_icd10_desc):
                print(filename)
                file_path = os.path.join(path_to_input, filename)
                print(f"counter is: {counter}")
                print(f"Processing file: {file_path}")
                note_df = pd.read_csv(file_path, sep="|")
                note_df_with_dial= dial_generation_functions.generate_dialouge_for_df(note_df= note_df,
                                                                                    openai_client = openai_client,
                                                                                    )
                
                dial_generation_functions.save_df_with_dial(df= note_df_with_dial, 
                                                            path= file_path)
                counter += 1

if __name__ == '__main__':
    main()