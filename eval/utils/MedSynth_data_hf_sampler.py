from datasets import load_dataset
from huggingface_hub import HfFolder
import pandas as pd
from datetime import datetime

from utils import constants

HfFolder.save_token(constants.HF_WRITE_TOKEN)



def get_sample_AhmadData(num_samples, 
                         dataset_path_hf= "Ahmad0067/SynthDataGen_llama3_Dial2Note_Ahmad_whole_data_instruct_dataset2024-08-08_13-11"):
    
    dataset = load_dataset(dataset_path_hf, split='train')

    # We shufle it everytime, with no seed. This is because the disase are different and
    # we don't want to be misled by a subset of disease.
    shuffled_dataset = dataset.shuffle() 
    sampled_data = shuffled_dataset.take(num_samples)

    # Convert to a DataFrame
    df = pd.DataFrame(sampled_data)

    # Rename columns to match required names by instruct_datasets.py 
    # Note that NoteChat doesnt have SOAP format. I just used this name for instruct_datasets.py 
    #df.rename(columns={'data': 'note', "conversation":"dialogue"}, inplace=True)

    return df


def save_AhmadData_sample_to_csv(df, num_samples,
                                 base_name= "AhmadData",
                                 path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/AhmadData_Sample"):
                
                current_date= datetime.now().strftime("%Y-%m-%d_%H-%M")
                full_name = f"Dial2Note_{base_name}_{num_samples}_{current_date}"

                full_path = f"{path}/{full_name}.csv"
                df.to_csv(full_path, index=False, sep= "|")
