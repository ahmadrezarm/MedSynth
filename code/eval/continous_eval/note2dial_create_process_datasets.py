#source: https://mlops.community/budget-instruction-fine-tuning-of-llama-3-8b-instructon-medical-data-with-hugging-face-google-colab-and-unsloth/
# source github (more update): https://github.com/Shekswess/LLM-Medical-Finetuning/blob/main/src/data_processing/create_process_datasets.py

import logging
import os
from datetime import datetime


import pandas as pd
from datasets import Dataset, DatasetDict
from utils.note2dial_instruct_datasets import (
    Llama3InstructDataset,
)
from utils import constants

from huggingface_hub import HfFolder
HfFolder.save_token(constants.HF_WRITE_TOKEN)

REMOVE_COLUMNS = []
RENAME_COLUMNS = {} #{"Polished Note":"note", "polished_dial": "dialogue", "Note": "unpolished_note", "dial": "unpolished_dialogue"} #
INSTRUCTION = constants.dial_augmentor_system_prompt

# edit this whenevr you wanna make a new dataset
#Ahmad57_data_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/AhmadData_Sample/Dial2Note_AhmadData_57_2024-12-14_11-19.csv"
Primock_data_path= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/eval/continous_eval/PriMock57/unified_data/primock57.csv"

current_date= datetime.now().strftime("%Y-%m-%d_%H-%M")

repo_name= f"SynthDataGen_llama3_Note2Dial_ACI_PriMock57_Only_instruct_dataset{current_date}" #

DATASETS_PATHS = [ Primock_data_path, ] #,  , Ahmad_whole_data_path, constants.Aci_train_path,
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def process_dataset(dataset_path: str, model: str) -> pd.DataFrame:
    """
    Process the instruct dataset to be in the format required by the model.
    :param dataset_path: The path to the dataset.
    :param model: The model to process the dataset for.
    :return: The processed dataset.
    """
    logger.info(f"Processing dataset: {dataset_path} for {model} instruct model.")
    # if model == "gemma":
    #     dataset = GemmaInstructDataset(dataset_path)
    # elif model == "mistral":
    #     dataset = MistralInstructDataset(dataset_path)
    # elif model == "llama":
    #     dataset = LlamaInstructDataset(dataset_path)
    if model == "llama3":
        dataset = Llama3InstructDataset(dataset_path)
    else:
        raise ValueError(f"Model {model} not supported!")
    
    if REMOVE_COLUMNS:
        dataset.drop_columns(REMOVE_COLUMNS)
        logger.info("Columns removed!")

    if RENAME_COLUMNS:
        dataset.rename_columns(RENAME_COLUMNS)
        logger.info("Columns renamed!")

    dataset.create_instruction(INSTRUCTION)
    logger.info("Instructions created!")
    #dataset.drop_bad_rows(["input", "output"])
    #logger.info("Bad rows dropped!")
    dataset.create_prompt()
    logger.info("Prompt column created!")
    return dataset.get_dataset()


def create_dataset_hf(
    dataset: pd.DataFrame,
) -> DatasetDict:
    """
    Create a Hugging Face dataset from the pandas dataframe.
    :param dataset: The pandas dataframe.
    :return: The Hugging Face dataset.
    """
    dataset.reset_index(drop=True, inplace=True)
    return DatasetDict({"train": Dataset.from_pandas(dataset)})


if __name__ == "__main__":
    #processed_data_path = r"/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval/instruction_tuning_data"
    #os.makedirs(processed_data_path, exist_ok=True)

    mistral_datasets = []
    gemma_datasets = []
    llama_datasets = []
    llama3_datasets = []
    for dataset_path in DATASETS_PATHS:
        print(f"{dataset_path}##############################")
        dataset_name = dataset_path.split(os.sep)[-1].split(".")[0]

        llama3_dataset = process_dataset(dataset_path, "llama3")
        llama3_datasets.append(llama3_dataset)
        
        # I commented the below lines right before adding note_chat data
        #llama3_dataset = create_dataset_hf(llama3_dataset)
        #llama3_dataset.push_to_hub(f"llama3_{dataset_name}_instruct_dataset_v3", private=True)

    llama3_dataset = pd.concat(llama3_datasets, ignore_index=True)
    llama3_dataset = create_dataset_hf(llama3_dataset)

    #llama3_dataset.save_to_disk(
    #    os.path.join(processed_data_path, f"llama3_instruct_{DATASETS_PATHS[-1][-10:]}_dataset")
    #)
    llama3_dataset.push_to_hub(repo_name, private= True)

