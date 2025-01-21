import openai
import re
import pandas as pd
from datetime import datetime
import random

from . import gen_constants 

# make the openai key an env variable and load it here
def initialize_openai_client():
    """
    Initialize the OpenAI client with an API key read from a file.

    Args:
    api_key_file (str): The path to the file containing the OpenAI API key.

    Returns:
    openai.Client: An instance of the OpenAI client.
    """
    try:
        client = openai.Client(api_key= "sk-proj-6jNq4RERBKwdpFStA2mFT3BlbkFJdCWfVwfvNqkiwaR48VAm") #gen_constants.TCAIREM_OPENAI_API_KEY
        print("OpenAI client initialized successfully!")
        return client
    except Exception as e:
        print(f"An error occurred in initializing OpenAI API: {e}")
        return None
    



def generate_dialogue(note, openai_client):
        aci_train_df= pd.read_csv(gen_constants.ACI_TRAIN_SET_PATH)
        randome_index_1= random.randint(32, 66) # just aci
        randome_index_2= random.randint(32, 66) # just aci
        randome_index_3= random.randint(32, 66) # just aci

        aci_note_sample_1= aci_train_df["dialogue"][randome_index_1]
        aci_dial_sample_1= aci_train_df["note"][randome_index_1]

        aci_note_sample_2= aci_train_df["dialogue"][randome_index_2]
        aci_dial_sample_2= aci_train_df["note"][randome_index_2]

        aci_note_sample_3= aci_train_df["dialogue"][randome_index_3]
        aci_dial_sample_3= aci_train_df["note"][randome_index_3]


        dial_prompt= gen_constants.DIALOGUE_GENERATOR_SYSTEM_PROMPT.format(EXAMPLE_1_NOTE=aci_note_sample_1,
                                                                           EXAMPLE_1_DIALOGUE= aci_dial_sample_1,
                                                                           EXAMPLE_2_NOTE= aci_note_sample_2,
                                                                           EXAMPLE_2_DIALOGUE= aci_dial_sample_2,
                                                                           EXAMPLE_3_NOTE= aci_note_sample_3,
                                                                           EXAMPLE_3_DIALOGUE= aci_dial_sample_3)
        
        note_response = openai_client.chat.completions.create(
                model=gen_constants.dial_generator_config["model"],
                temperature = gen_constants.dial_generator_config["temperature"],
                max_tokens = gen_constants.dial_generator_config["max_tokens"],
                top_p = gen_constants.dial_generator_config["top_p"],
                frequency_penalty = gen_constants.dial_generator_config["frequency_penalty"],
                presence_penalty = gen_constants.dial_generator_config["presence_penalty"],
                messages=[
                    {
                        "role": "system",
                        "content": dial_prompt
                    },
                    {
                        "role": "user",
                        "content": note
                    }
                ]
                )
        return note_response.choices[0].message.content


def polish_dialogue(dialogue, medical_not, openai_client):
        
        dial_polish_prompt= gen_constants.DIALOGUE_POLISHER_SYSTEM_PROMPT.format(MEDICAL_NOTE= medical_not)
        note_response = openai_client.chat.completions.create(
                model=gen_constants.dial_polisher_config["model"],
                temperature = gen_constants.dial_polisher_config["temperature"],
                max_tokens = gen_constants.dial_polisher_config["max_tokens"],
                top_p = gen_constants.dial_polisher_config["top_p"],
                frequency_penalty = gen_constants.dial_polisher_config["frequency_penalty"],
                presence_penalty = gen_constants.dial_polisher_config["presence_penalty"],
                messages=[
                    {
                        "role": "system",
                        "content": dial_polish_prompt
                    },
                    {
                        "role": "user",
                        "content": dialogue
                    }
                ]
                )
        return note_response.choices[0].message.content



def generate_dialouge_for_df(note_df, openai_client):
    filtered_df = note_df[note_df["Note"] != "Rejected"]
    for idx, row in filtered_df.iterrows():
        print(f"Generating dialogue for idx: {idx}")
        dial = generate_dialogue(row["Note"], openai_client=openai_client)
        polished_dial = polish_dialogue(dialogue=dial, medical_not= row["Note"], openai_client=openai_client)
        note_df.at[idx, "dial"] = dial
        note_df.at[idx, "polished_dial"] = polished_dial

    return note_df



def save_df_with_dial(df, path):
     df.to_csv(f"{path}_with_dial.csv", sep="|", index= False)

