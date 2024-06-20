import os
import pandas as pd

from gen_utils import dial_generation_functions
#from .gen_utils import gen_constants


path_to_input= "/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/output/notes_onVector/Phase_2/"

def main():
    openai_client= dial_generation_functions.initialize_openai_client()
    for filename in os.listdir(path_to_input):
        if filename.endswith(".csv"):
            file_path = os.path.join(path_to_input, filename)
            print(f"Processing file: {file_path}")
            note_df = pd.read_csv(file_path, sep="|")
            note_df_with_dial= dial_generation_functions.generate_dialouge_for_df(note_df= note_df,
                                                                                  openai_client= openai_client)
            
            dial_generation_functions.save_df_with_dial(df= note_df_with_dial, 
                                                        path= file_path)


if __name__ == '__main__':
    main()