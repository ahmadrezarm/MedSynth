import openai
import re
import pandas as pd
from datetime import datetime
import random
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from huggingface_hub import HfFolder
import pandas as pd
from datetime import datetime


HF_WRITE_TOKEN = os.getenv('MY_HF_WRITE_TOKEN')
HF_READ_TOKEN = os.getenv('MY_HF_READ_TOKEN')

HfFolder.save_token(HF_WRITE_TOKEN)

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
        client = openai.Client(api_key= gen_constants.OPENAI_API_KEY) #gen_constants.TCAIREM_OPENAI_API_KEY
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




############# Ablation all llama 3 begins ################ 
def load_local_model(MODEL_PATH):


    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )
 
    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            quantization_config=quantization_config,
            device_map="auto", 
        )
    return model, tokenizer


def generate_dialogue_lamma(note, model, tokenizer):
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


        dial_generator_system_prompt= gen_constants.DIALOGUE_GENERATOR_SYSTEM_PROMPT.format(EXAMPLE_1_NOTE=aci_note_sample_1,
                                                                           EXAMPLE_1_DIALOGUE= aci_dial_sample_1,
                                                                           EXAMPLE_2_NOTE= aci_note_sample_2,
                                                                           EXAMPLE_2_DIALOGUE= aci_dial_sample_2,
                                                                           EXAMPLE_3_NOTE= aci_note_sample_3,
                                                                           EXAMPLE_3_DIALOGUE= aci_dial_sample_3)
        
        pre_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            {dial_generator_system_prompt}<|eot_id|> <|start_header_id|>user<|end_header_id|>{note}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        prompt= pre_prompt 
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # I'm not sure if presence_penalty and frequency_penalty are impelemented with the same logic as OpenAI, and also I used 0 for GPT model. So, I'm skipping them for Llama.
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_constants.dial_generator_config["max_tokens"], 
                temperature= gen_constants.dial_generator_config["temperature"], 
                top_p= gen_constants.dial_generator_config["top_p"]
            )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        start_index = response[0].rfind("<|start_header_id|>assistant<|end_header_id|>") + 45
        #### added -10 to prevent the <|end_header_id|> to get passed back
        dial = response[0][start_index:-10].strip()

        return dial

 

def polish_dialogue_lamma(dialogue, medical_not, model, tokenizer):
        
        dial_polish__system_prompt= gen_constants.DIALOGUE_POLISHER_SYSTEM_PROMPT.format(MEDICAL_NOTE= medical_not)

        pre_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            {dial_polish__system_prompt}<|eot_id|> <|start_header_id|>user<|end_header_id|>{dialogue}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        prompt= pre_prompt 
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # I'm not sure if presence_penalty and frequency_penalty are impelemented with the same logic as OpenAI, and also I used 0 for GPT model. So, I'm skipping them for Llama.
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_constants.dial_polisher_config["max_tokens"], 
                temperature= gen_constants.dial_polisher_config["temperature"], 
                top_p= gen_constants.dial_polisher_config["top_p"]
            )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        start_index = response[0].rfind("<|start_header_id|>assistant<|end_header_id|>") + 45
        #### added -10 to prevent the <|end_header_id|> to get passed back
        polished_dial = response[0][start_index:-10].strip()

        return polished_dial


def generate_dialouge_for_df_lamma(note_df, model, tokenizer):
    filtered_df = note_df[note_df["Note"] != "Rejected"]
    for idx, row in filtered_df.iterrows():
        print(f"Generating dialogue for idx: {idx}")
        dial = generate_dialogue_lamma(row["Note"], model= model, tokenizer= tokenizer)
        polished_dial = polish_dialogue_lamma(dialogue=dial, medical_not= row["Note"], model = model, tokenizer=tokenizer)
        note_df.at[idx, "dial"] = dial
        note_df.at[idx, "polished_dial"] = polished_dial

    return note_df


############# Ablation all llama 3 ends ################ 






############# Ablation all Qwen begins ################ 

def generate_dialogue_qwen(note, model, tokenizer):
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


        dial_generator_system_prompt= gen_constants.DIALOGUE_GENERATOR_SYSTEM_PROMPT.format(EXAMPLE_1_NOTE=aci_note_sample_1,
                                                                           EXAMPLE_1_DIALOGUE= aci_dial_sample_1,
                                                                           EXAMPLE_2_NOTE= aci_note_sample_2,
                                                                           EXAMPLE_2_DIALOGUE= aci_dial_sample_2,
                                                                           EXAMPLE_3_NOTE= aci_note_sample_3,
                                                                           EXAMPLE_3_DIALOGUE= aci_dial_sample_3)

        pre_prompt = [
            {"role": "system", "content": dial_generator_system_prompt},
            {"role": "user", "content": note}
        ]
        prompt= pre_prompt 

        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=gen_constants.dial_polisher_config["max_tokens"], 
                temperature= gen_constants.dial_polisher_config["temperature"], 
                top_p= gen_constants.dial_polisher_config["top_p"]
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        polished_dial = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return polished_dial





def polish_dialogue_qwen(dialogue, medical_not, model, tokenizer):
    dial_polish__system_prompt= gen_constants.DIALOGUE_POLISHER_SYSTEM_PROMPT.format(MEDICAL_NOTE= medical_not)
    pre_prompt = [
        {"role": "system", "content": dial_polish__system_prompt},
        {"role": "user", "content": dialogue}
    ]
    prompt= pre_prompt 

    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=gen_constants.note_polisher_config["max_tokens"], 
            temperature= 0.001, #gen_constants.note_generator_config["temperature"], 
            top_p= gen_constants.note_polisher_config["top_p"]
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    polished_note = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return polished_note



def generate_dialouge_for_df_qwen(note_df, model, tokenizer):
    filtered_df = note_df[note_df["Note"] != "Rejected"]
    for idx, row in filtered_df.iterrows():
        print(f"Generating dialogue for idx: {idx}")
        dial = generate_dialogue_qwen(row["Note"], model= model, tokenizer= tokenizer)
        polished_dial = polish_dialogue_qwen(dialogue=dial, medical_not= row["Note"], model = model, tokenizer=tokenizer)
        note_df.at[idx, "dial"] = dial
        note_df.at[idx, "polished_dial"] = polished_dial

    return note_df
############# Ablation all Qwen ends ################ 

