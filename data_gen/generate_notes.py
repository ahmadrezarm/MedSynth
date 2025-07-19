import pandas as pd
import os
import argparse

from gen_utils import note_generation_functions
from gen_utils import gen_constants

def file_exists(disease_description, path_to_save_notes):
    """Check if a file for the disease description already exists in the directory."""
    for filename in os.listdir(path_to_save_notes):
        if disease_description in filename and filename.endswith('.csv'):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical notes.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save generated notes")
    parser.add_argument("--num_icd10", type=int, default=50, help="Number of ICD10 descriptions to use")
    parser.add_argument("--notes_per_icd10", type=int, default=5, help="Number of notes to generate per ICD10 description")
    args = parser.parse_args()
    path_to_save_notes = args.output_dir

    openai_client= note_generation_functions.initialize_openai_client()

    input_csv_path = os.path.join(os.path.dirname(__file__), "../../data/input/IQVIA/IQVIA_cleaned.csv")
    df = pd.read_csv(input_csv_path, sep="|")
    top_n_icd10_desc = df['ICD10_desc'].head(args.num_icd10).tolist()

    counter= 1
    for disease in top_n_icd10_desc:
            print(f"counter is: {counter}")
            print(disease)
            if not file_exists(disease, path_to_save_notes):
                note_generation_functions.generate_and_save_medical_notes(disease_description= disease, 
                                                              notes_count= args.notes_per_icd10, 
                                                              openai_client= openai_client,
                                                              path_to_save_notes= path_to_save_notes) 
            counter += 1


if __name__ == '__main__':
    main()