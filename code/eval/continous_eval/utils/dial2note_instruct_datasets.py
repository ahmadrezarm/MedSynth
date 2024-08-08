#source: https://mlops.community/budget-instruction-fine-tuning-of-llama-3-8b-instructon-medical-data-with-hugging-face-google-colab-and-unsloth/
# source github (more update): https://github.com/Shekswess/LLM-Medical-Finetuning/blob/main/src/data_processing/create_process_datasets.py
from abc import ABC, abstractmethod

import pandas as pd


class InstructDataset(ABC):
    """
    Abstract class for creating Instruct Datasets
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the dataset
        :param dataset_path: The path to the dataset
        """
        self.dataset = None
        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str) -> None:
        """
        Load the dataset from the given path
        :param dataset_path: The path to the dataset
        :return: None
        """
        if dataset_path == "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/output/notes_onVector/Phase_2_500/500_sample/_combined_with_dial.csv":
                self.dataset = pd.read_csv(dataset_path, sep= "|")
        elif dataset_path == "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/output/notes_onVector/Phase_3_1k/with_dial/_combined_with_dial.csv":
                self.dataset = pd.read_csv(dataset_path, sep= "|")
        elif dataset_path == "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/output/notes_onVector/phase_4_250/_combined_with_dial.csv":
                self.dataset = pd.read_csv(dataset_path, sep= "|")
        elif dataset_path == "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/output/notes_onVector/phase_5_250/_combined_with_dial.csv":
                self.dataset = pd.read_csv(dataset_path, sep= "|")
        elif dataset_path == "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/output/notes_onVector/phase_6_500/_combined_with_dial.csv":
                self.dataset = pd.read_csv(dataset_path, sep= "|")
        elif dataset_path == "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/output/notes_onVector/phase_7_1.5k/_combined_with_dial.csv":
                self.dataset = pd.read_csv(dataset_path, sep= "|")

        else:
            self.dataset = pd.read_csv(dataset_path)

    def rename_columns(self, columns: dict[str, str]) -> None:
        """
        Rename the columns of the dataset
        :param columns: A dictionary of the form {old_name: new_name}
        :return: None
        """
        self.dataset = self.dataset.rename(columns=columns)

    def drop_columns(self, columns: list[str]) -> None:
        """
        Drop the columns from the dataset
        :param columns: A list of column names to drop
        :return: None
        """
        drop_columns = [col for col in columns if col in self.dataset.columns]
        self.dataset = self.dataset.drop(columns=drop_columns)

    def drop_bad_rows(self, columns: list[str]) -> None:
        """
        Drop the rows which have bad values in the columns
        :param columns: A list of columns to check for bad values
        :return: None
        """
        self.dataset = self.dataset.dropna(subset=columns)
        self.dataset = self.dataset.drop_duplicates(subset=columns)

    def create_instruction(self, instruction: str) -> None:
        """
        Create an instruction column in the dataset
        :param instruction: The instruction to add to the dataset
        :return: None
        """
        self.dataset["instruction"] = instruction

    @abstractmethod
    def create_prompt(self) -> None:
        """
        Create the prompt column in the dataset
        :return: None
        """
        pass

    def get_dataset(self) -> pd.DataFrame:
        """
        Get the dataset
        :return: The dataset
        """
        return self.dataset

# needs investigation to ensure correctness.
class MistralInstructDataset(InstructDataset):

    def create_prompt(self):
        """
        Create the prompt column in the dataset which will be used for
        """
        prompts = []
        for index, row in self.dataset.iterrows():
            prompt = f"""<s>[INST] {row['instruction']} This is the conversation: {row['dialogue']} [/INST] \\n {row['note_SOAP']}</s>"""
            prompts.append(prompt)
        self.dataset["prompt"] = prompts


# needs investigation to ensure correctness.
class LlamaInstructDataset(InstructDataset):
    def create_prompt(self):
        """
        Create the prompt column in the dataset which will be used for
        """
        prompts = []
        for index, row in self.dataset.iterrows():
            prompt = f"""[s][INST] {row['instruction']} This is the conversation: {row['dialogue']} [/INST] \\n {row['note_SOAP']}[/s]"""
            prompts.append(prompt)
        self.dataset["prompt"] = prompts



class Llama3InstructDataset(InstructDataset):
    # source for my edit: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/14
    
    def create_prompt(self):
        """
        Create the prompt column in the dataset which will be used for
        """
        prompts = []
        for index, row in self.dataset.iterrows():
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \n\n {{{{ {row['instruction']} }}}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n  {{{{ This is the conversation: {row['dialogue']} }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n {{{{ {row['note']} }}}}<|eot_id|>"""
            prompts.append(prompt)
            if index==0:
                print(prompt)
        self.dataset["prompt"] = prompts


 # needs investigation to ensure correctness.
class GemmaInstructDataset(InstructDataset):
    def create_prompt(self):
        """
        Create the prompt column in the dataset which will be used for
        """
        prompts = []
        for index, row in self.dataset.iterrows():
            prompt = f"<start_of_turn>user {row['instruction']} This is the conversation: {row['dialogue']}<end_of_turn> \\n <start_of_turn>model {row['note_SOAP']}<end_of_turn>model"
            prompts.append(prompt)
        self.dataset["prompt"] = prompts