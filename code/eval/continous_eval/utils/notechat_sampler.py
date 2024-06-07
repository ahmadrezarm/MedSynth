from datasets import load_dataset
from huggingface_hub import HfFolder
import pandas as pd
from datetime import datetime

from utils import constants

HfFolder.save_token(constants.HF_WRITE_TOKEN)



def get_sample_note_chat(dataset_path_hf= constants.NOTE_CHAT_HF_PATH, 
                         num_samples= constants.NUM_NOTE_CHAT_SAMPLES):
    
    dataset = load_dataset(dataset_path_hf, split='train')

    # We shufle it everytime, with no seed. This is because the disase are different and
    # we don't want to be misled by a subset of disease.
    shuffled_dataset = dataset.shuffle() 
    sampled_data = shuffled_dataset.take(num_samples)

    # Convert to a DataFrame
    df = pd.DataFrame(sampled_data)

    # Rename columns to match required names by instruct_datasets.py 
    # Note that NoteChat doesnt have SOAP format. I just used this name for instruct_datasets.py 
    df.rename(columns={'data': 'note', "conversation":"dialogue"}, inplace=True)

    return df


def save_note_chat_sample_to_csv(df, 
                                 base_name= constants.NOTE_CHAT_SAMPLE_BASE_NAME,
                                 path= constants.PATH_TO_SAVE_NOTE_CHAT_SAMPLES):
                
                current_date= datetime.now().strftime("%Y-%m-%d_%H-%M")
                full_name = f"{base_name}_{current_date}"

                full_path = f"{path}/{full_name}.csv"
                df.to_csv(full_path, index=False)