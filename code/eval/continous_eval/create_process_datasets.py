#source: https://mlops.community/budget-instruction-fine-tuning-of-llama-3-8b-instructon-medical-data-with-hugging-face-google-colab-and-unsloth/
# source github (more update): https://github.com/Shekswess/LLM-Medical-Finetuning/blob/main/src/data_processing/create_process_datasets.py

import logging
import os


import pandas as pd
from datasets import Dataset, DatasetDict
from utils.instruct_datasets import (
    GemmaInstructDataset,
    MistralInstructDataset,
    LlamaInstructDataset,
    Llama3InstructDataset,
)
from utils import constants

from huggingface_hub import HfFolder
HfFolder.save_token(constants.HF_WRITE_TOKEN)

REMOVE_COLUMNS = []
RENAME_COLUMNS = {}
INSTRUCTION = constants.summarizer_system_prompt
DATASETS_PATHS = [constants.Aci_train_path]

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
    if model == "gemma":
        dataset = GemmaInstructDataset(dataset_path)
    elif model == "mistral":
        dataset = MistralInstructDataset(dataset_path)
    elif model == "llama":
        dataset = LlamaInstructDataset(dataset_path)
    elif model == "llama3":
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
        dataset_name = dataset_path.split(os.sep)[-1].split(".")[0]

        llama3_dataset = process_dataset(dataset_path, "llama3")
        llama3_datasets.append(llama3_dataset)
        llama3_dataset = create_dataset_hf(llama3_dataset)
        llama3_dataset.push_to_hub(f"llama3_{dataset_name}_instruct_dataset_v3", private=True)

    llama3_dataset = pd.concat(llama3_datasets, ignore_index=True)
    llama3_dataset = create_dataset_hf(llama3_dataset)

    #llama3_dataset.save_to_disk(
    #    os.path.join(processed_data_path, f"llama3_instruct_{DATASETS_PATHS[-1][-10:]}_dataset")
    #)

    llama3_dataset.push_to_hub(f"llama3_{DATASETS_PATHS[-1][:-10]}_instruct_dataset_v3", private= True)

