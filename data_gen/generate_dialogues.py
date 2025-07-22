import os
import pandas as pd
import argparse
from pathlib import Path


from gen_utils import dial_generation_functions

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical dialogues.")
    parser.add_argument("--path_to_input", type=str, required=True, help="Path to the directory you have notes in")
    parser.add_argument("--aci_train_path", type=str, default=None, help="Path to TaskC-TrainingSet.csv")
    args = parser.parse_args()

    aci_train_path = Path(args.aci_train_path).expanduser().resolve()


    counter= 1
    openai_client= dial_generation_functions.initialize_openai_client()
    for filename in os.listdir(args.path_to_input):
        if filename.endswith(".csv"):
                print(filename)
                file_path = os.path.join(args.path_to_input, filename)
                print(f"counter is: {counter}")
                print(f"Processing file: {file_path}")
                note_df = pd.read_csv(file_path, sep="|")
                note_df_with_dial= dial_generation_functions.generate_dialouge_for_df(note_df= note_df,
                                                                                    openai_client = openai_client,
                                                                                     ACI_TRAIN_SET_PATH= aci_train_path)
                
                dial_generation_functions.save_df_with_dial(df= note_df_with_dial, 
                                                            path= file_path)
                counter += 1

if __name__ == '__main__':
    main()