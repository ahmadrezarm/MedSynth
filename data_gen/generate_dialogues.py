import os
import pandas as pd
import argparse

from gen_utils import dial_generation_functions

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical dialogues.")
    parser.add_argument("--path_to_input", type=str, required=True, help="Path to you have your notes in")
    parser.add_argument("--num_icd10", type=int, default=50, help="Number of ICD10 descriptions to use")
    args = parser.parse_args()

    input_csv_path = os.path.join(os.path.dirname(__file__), "../../data/input/IQVIA/IQVIA_cleaned.csv")
    df = pd.read_csv(input_csv_path, sep="|")

    top_n_icd10_desc = df['ICD10_desc'].head(args.num_icd10).tolist()

    counter= 1
    openai_client= dial_generation_functions.initialize_openai_client()
    for filename in os.listdir(args.path_to_input):
        if filename.endswith(".csv"):
            if any(desc in filename for desc in top_n_icd10_desc):
                print(filename)
                file_path = os.path.join(args.path_to_input, filename)
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