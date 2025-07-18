
import openai
import os
import re
import pandas as pd
from datetime import datetime
import random

from . import gen_constants


############## For ablations begin #################
from huggingface_hub import HfFolder
import pandas as pd
from datetime import datetime

from unsloth import FastLanguageModel

HF_WRITE_TOKEN = os.getenv('MY_HF_WRITE_TOKEN')
HF_READ_TOKEN = os.getenv('MY_HF_READ_TOKEN')

HfFolder.save_token(HF_WRITE_TOKEN)

############## For ablations end #################


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
        client = openai.Client(api_key= gen_constants.OPENAI_API_KEY)
        print("OpenAI client initialized successfully!")
        return client
    except Exception as e:
        print(f"An error occurred in initializing OpenAI API: {e}")
        return None
    



def doctor_generate_scenario(condition, scenario_provider_memory, openai_client):

        aci_train_df= pd.read_csv(gen_constants.ACI_TRAIN_SET_PATH)
        randome_index= random.randint(0, 66)
        aci_note_sample= aci_train_df["note"][randome_index]
        scenario_prompt= gen_constants.SCENARIO_PROVIDER_SYSTEM_PROMPT.format(EXAMPLE_NOTE=aci_note_sample)


        scenario_response = openai_client.chat.completions.create(
                model=gen_constants.scenario_generator_config["model"],
                temperature = gen_constants.scenario_generator_config["temperature"],
                max_tokens = gen_constants.scenario_generator_config["max_tokens"],
                top_p = gen_constants.scenario_generator_config["top_p"],
                frequency_penalty = gen_constants.scenario_generator_config["frequency_penalty"],
                presence_penalty = gen_constants.scenario_generator_config["presence_penalty"],
                messages=[
                    {
                        "role": "system",
                        "content": scenario_prompt #doctor_scenario_generator_system_prompt
                    },
                    {
                        "role": "user",
                        "content": condition
                    }
                    ]+ scenario_provider_memory
                )
        print(f"len of scenario_provider_memory is: {len(scenario_provider_memory)}")
        #print("##############",scenario_provider_memory)

        return scenario_response.choices[0].message.content



################### For ablation of No Judge begin #################
 
def judge_evaluate_scenario(scenario, judge_conversations_memory, openai_client):
    # Add system prompt and user scenario to memory
        judge_conversations_memory += [
        {"role": "user", "content": scenario}
        ]
        evaluation_response = openai_client.chat.completions.create(
                model=gen_constants.scenario_judge_config["model"],
                temperature = gen_constants.scenario_judge_config["temperature"],
                max_tokens = gen_constants.scenario_judge_config["max_tokens"],
                top_p = gen_constants.scenario_judge_config["top_p"],
                frequency_penalty = gen_constants.scenario_judge_config["frequency_penalty"],
                presence_penalty = gen_constants.scenario_judge_config["presence_penalty"],
                messages=judge_conversations_memory
                )
        # Accessing the last message's content correctly
        latest_message = evaluation_response.choices[0].message.content

        print(latest_message)
        #print("Latest message in judge is: ", latest_message)
        decision = latest_message.split()[-1]  # Extract the last word

        # Update memory with the model's latest response
        judge_conversations_memory.append({"role": "assistant", "content": latest_message})
        print(f"decission is: {decision}")
        #print(f"latest_message is: {latest_message}")
        #print(judge_conversations_memory)
        print("len of judge_conversation_momory is: ",len(judge_conversations_memory))
        return decision, latest_message



def judge_evaluate_scenario_no_judge_ablation(scenario, judge_conversations_memory, openai_client):
     decision = "Go"
     latest_message = "Go"
     return decision, latest_message




""" 
def abbreviate_note(note, openai_client):
        note_response = openai_client.chat.completions.create(
                model=gen_constants.note_abbreviator_config["model"],
                temperature = gen_constants.note_abbreviator_config["temperature"],
                max_tokens = gen_constants.note_abbreviator_config["max_tokens"],
                top_p = gen_constants.note_abbreviator_config["top_p"],
                frequency_penalty = gen_constants.note_abbreviator_config["frequency_penalty"],
                presence_penalty = gen_constants.note_abbreviator_config["presence_penalty"],
                messages=[
                    {
                        "role": "system",
                        "content": gen_constants.NOTE_ABBREVIATOR_SYSTEM_PROMPT 
                    },
                    {
                        "role": "user",
                        "content": note
                    }
                ]
                )
        return note_response.choices[0].message.content
    """

def abbreviate_note_judge_ablation(note, openai_client):
     return "ablation"

################# For ablation No Judge end #####################




################ For ablation Llama 3.3 as Judge begins ############
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from huggingface_hub import HfFolder
import os

from unsloth import FastLanguageModel

HF_WRITE_TOKEN = os.getenv('MY_HF_WRITE_TOKEN')
HF_READ_TOKEN = os.getenv('MY_HF_READ_TOKEN')

HfFolder.save_token(HF_WRITE_TOKEN)

from gen_utils import gen_constants

MODEL_PATH = f"/model-weights/Llama-3.3-70B-Instruct"


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


def load_unsloth_model(model_id):
    model, tokenizer=  FastLanguageModel.from_pretrained(model_name = model,
                                                                        max_seq_length = gen_constants.scenario_judge_config["max_tokens"],
                                                                        dtype = torch.bfloat16,
                                                                        load_in_4bit = False,
                                                                        temperature= 0.001)

    ############# add this to the next function instead ######
    FastLanguageModel.for_inference(model)
    ##########################################################
    return model, tokenizer



         

def judge_evaluate_scenario_with_llama(scenario, judge_conversations_memory, model, tokenizer):
    """
    Evaluates a scenario using a custom model and returns the decision and response.
    
    Args:
        scenario (str): The scenario to evaluate.
        judge_conversations_memory (list): The conversation memory to maintain context.
        model: The model used for generation.
        tokenizer: The tokenizer used for encoding and decoding.
        generation_config (dict): Configuration for generation parameters.

    Returns:
        tuple: The decision (str) and the latest model response (str).
    """
    # Add user scenario to the memory
    judge_conversations_memory.append({"role": "user", "content": scenario})

    # Combine all user and assistant messages into one formatted query
    conversation_history = "\n".join(
        [f"<|start_header_id|>{entry['role']}<|end_header_id|> {entry['content']} <|eot_id|>" for entry in judge_conversations_memory]
    )

    # Use the prompt template
    pre_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            {gen_constants.SCENARIO_JUDGE_SYSTEM_PROMPT}<|eot_id|>
        """
    
    prompt= pre_prompt +  conversation_history

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # I'm not sure if presence_penalty and frequency_penalty are impelemented with the same logic as OpenAI, and also I used 0 for GPT model. So, I'm skipping them for Llama.
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_constants.scenario_judge_config["max_tokens"], #,
            temperature= 0.001, #gen_constants.scenario_judge_config["temperature"]: Cannot be 0 in tranformers
            top_p= gen_constants.scenario_judge_config["top_p"]
        )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    start_index = response[0].rfind("<|start_header_id|>assistant<|end_header_id|>") + 45
    latest_message = response[0][start_index:].strip()

    # Extract the decision (e.g., the last word or a specific format)
    decision = latest_message.split()[-1]

    # Update the memory with the assistant's response
    judge_conversations_memory.append({"role": "assistant", "content": latest_message})

    # Debugging logs (optional)
    print(f"Latest message: {latest_message}")
    print(f"Decision: {decision}")
    print(f"Updated conversation memory length: {len(judge_conversations_memory)}")

    return decision, latest_message





def generate_and_save_medical_notes_llama3_as_judge(disease_description, notes_count, openai_client,
                                                    model, tokenizer,
                                                    path_to_save_notes):
    
    approved_notes = []
    rejected_scenarios = []
    judge_conversations_memory = []
    scenario_provider_memory= []
    try:
        while True:
            scenario = doctor_generate_scenario(disease_description, scenario_provider_memory, openai_client)

            # if len(scenario_provider_memory) == 0:
            #     scenario_provider_memory.append({"role": "user", "content": disease_description})
                
            scenario_provider_memory.append({"role": "assistant", "content": scenario})

            decision, latest_message = judge_evaluate_scenario_with_llama(scenario, judge_conversations_memory, model, tokenizer)
            print(decision)
            print(latest_message)
            if decision == "Go<|eot_id|>" or decision == "Go.<|eot_id|>":
                role = _extract_role(scenario)
                note = doctor_generate_note(scenario, openai_client)
                polished_note = polish_note(note, openai_client)
                abbreviated_note= abbreviate_note_judge_ablation(polished_note, openai_client)
                approved_notes.append({"Disease Description": disease_description, "Scenario": scenario, "Note": note, "Polished Note": polished_note, "Abbreviated Note": abbreviated_note, "Role": role })
                scenario_provider_memory = []
                print(f"Note number {len(approved_notes)} has been generated!")
            else:
                rejected_scenarios.append({"Disease Description": disease_description, "Scenario": scenario, "Note": "Rejected", "Polished Note": "Rejected", "Abbreviated Note": "Rejected", "Role": "Rejected"})
                # to save on input tokens: drop the rejected scenario from the memory
                del judge_conversations_memory[-2:]

                scenario_provider_memory.append({"role": "user", "content": latest_message})

            # reset to respect the input token limit
            # the second condition prevents the occurance of infinite loops of rejecting scenarios.
            if  ((len(approved_notes) % 4) == 0) or (len(scenario_provider_memory) >= 6):
                     judge_conversations_memory = []

            # Respecting the number of needed approved notes
            if len(approved_notes) >= notes_count:
                break

    except Exception as e:
        print(f"An error occurred during note generation: {e}")
    
    finally:
        # Combine the results
        results = approved_notes + rejected_scenarios
        current_date = datetime.now().strftime("%Y-%m-%d")
        sanitized_description = sanitize_filename(disease_description)
        results_df= pd.DataFrame(results)
        full_path= f"{path_to_save_notes}/{sanitized_description}_{current_date}.csv"
        results_df.to_csv(full_path, index=False, sep="|")




################ For ablation Llama 3.3 as Judge ends ############




################ For ablation whole pipeline with Llama 3.3 begins ############
""" 
We already have these functions from previous ablation we could use:
        1. load_local_model
        2. judge_evaluate_scenario_with_llama

We need a few more functions for note generation:
        1. doctor_generate_scenario_with_llama
        2. doctor_generate_note_with_lamma
        3. polish_note_with_lamma
        4. generate_and_save_medical_notes_all_llama
"""

def doctor_generate_scenario_with_llama(condition, scenario_provider_memory, model, tokenizer):
    aci_train_df= pd.read_csv(gen_constants.ACI_TRAIN_SET_PATH)
    randome_index= random.randint(0, 66)
    aci_note_sample= aci_train_df["note"][randome_index]

    scenario_system_prompt= gen_constants.SCENARIO_PROVIDER_SYSTEM_PROMPT.format(EXAMPLE_NOTE=aci_note_sample)
    # Use the prompt template
    pre_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            {scenario_system_prompt}<|eot_id|> <|start_header_id|>user<|end_header_id|>{condition}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    
    # Combine all user and assistant messages into one formatted query
    conversation_history = "\n".join(
        [f"<|start_header_id|>{entry['role']}<|end_header_id|> {entry['content']} <|eot_id|>" for entry in scenario_provider_memory]
    )
    
    prompt= pre_prompt +  conversation_history

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # I'm not sure if presence_penalty and frequency_penalty are impelemented with the same logic as OpenAI, and also I used 0 for GPT model. So, I'm skipping them for Llama.
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_constants.scenario_generator_config["max_tokens"], 
            temperature= gen_constants.scenario_generator_config["temperature"], 
            top_p= gen_constants.scenario_generator_config["top_p"]
        )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    start_index = response[0].rfind("<|start_header_id|>assistant<|end_header_id|>") + 45

    #### added -10 to prevent the <|end_header_id|> to get passed back
    scenario = response[0][start_index:-10].strip()

    print(f"len of scenario_provider_memory is: {len(scenario_provider_memory)}")

    return scenario



def doctor_generate_note_with_lamma(scenario, model, tokenizer):
    aci_train_df= pd.read_csv(gen_constants.ACI_TRAIN_SET_PATH)
    randome_index= random.randint(0, 66)
    aci_note_sample= aci_train_df["note"][randome_index]

    note_generator_system_prompt= gen_constants.NOTE_GENERATOR_SYSTEM_PROMPT.format(EXAMPLE_NOTE=aci_note_sample)  
    # Use the prompt template
    pre_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            {note_generator_system_prompt}<|eot_id|> <|start_header_id|>user<|end_header_id|>{scenario}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    
    prompt= pre_prompt 
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # I'm not sure if presence_penalty and frequency_penalty are impelemented with the same logic as OpenAI, and also I used 0 for GPT model. So, I'm skipping them for Llama.
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_constants.note_generator_config["max_tokens"], 
            temperature= gen_constants.note_generator_config["temperature"], 
            top_p= gen_constants.note_generator_config["top_p"]
        )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    start_index = response[0].rfind("<|start_header_id|>assistant<|end_header_id|>") + 45
    #### added -10 to prevent the <|end_header_id|> to get passed back
    note = response[0][start_index:-10].strip()

    return note



def polish_note_with_lamma(note, model, tokenizer):
    note_polisher_system_prompt= gen_constants.NOTE_POLISHER_SYSTEM_PROMPT 
    # Use the prompt template
    pre_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            {note_polisher_system_prompt}<|eot_id|> <|start_header_id|>user<|end_header_id|>{note}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    
    prompt= pre_prompt 
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # I'm not sure if presence_penalty and frequency_penalty are impelemented with the same logic as OpenAI, and also I used 0 for GPT model. So, I'm skipping them for Llama.
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_constants.note_polisher_config["max_tokens"], 
            temperature= 0.001, #gen_constants.note_generator_config["temperature"], 
            top_p= gen_constants.note_polisher_config["top_p"]
        )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    start_index = response[0].rfind("<|start_header_id|>assistant<|end_header_id|>") + 45
    #### added -10 to prevent the <|end_header_id|> to get passed back
    polished_note = response[0][start_index:-10].strip()

    return polished_note


def abbreviate_note_all_lamma():
     return "ablation"


def generate_and_save_medical_notes_all_llama3(disease_description, notes_count, 
                                                    model, tokenizer,
                                                    path_to_save_notes):
    
    approved_notes = []
    rejected_scenarios = []
    judge_conversations_memory = []
    scenario_provider_memory= []
    try:
        while True:
            scenario = doctor_generate_scenario_with_llama(disease_description, scenario_provider_memory, model, tokenizer)

            # if len(scenario_provider_memory) == 0:
            #     scenario_provider_memory.append({"role": "user", "content": disease_description})
                
            scenario_provider_memory.append({"role": "assistant", "content": scenario})

            decision, latest_message = judge_evaluate_scenario_with_llama(scenario, judge_conversations_memory, model, tokenizer)
            print(decision)
            print(latest_message)
            if decision == "Go<|eot_id|>" or decision == "Go.<|eot_id|>":
                role = _extract_role(scenario)
                note = doctor_generate_note_with_lamma(scenario, model, tokenizer)
                polished_note = polish_note_with_lamma(note, model, tokenizer)
                abbreviated_note= abbreviate_note_all_lamma()
                approved_notes.append({"Disease Description": disease_description, "Scenario": scenario, "Note": note, "Polished Note": polished_note, "Abbreviated Note": abbreviated_note, "Role": role })
                scenario_provider_memory = []
                print(f"Note number {len(approved_notes)} has been generated!")
            else:
                rejected_scenarios.append({"Disease Description": disease_description, "Scenario": scenario, "Note": "Rejected", "Polished Note": "Rejected", "Abbreviated Note": "Rejected", "Role": "Rejected"})
                # to save on input tokens: drop the rejected scenario from the memory
                del judge_conversations_memory[-2:]

                scenario_provider_memory.append({"role": "user", "content": latest_message})

            # reset to respect the input token limit
            # the second condition prevents the occurance of infinite loops of rejecting scenarios.
            if  ((len(approved_notes) % 4) == 0) or (len(scenario_provider_memory) >= 6):
                     judge_conversations_memory = []

            # Respecting the number of needed approved notes
            if len(approved_notes) >= notes_count:
                break

    except Exception as e:
        print(f"An error occurred during note generation: {e}")
    
    finally:
        # Combine the results
        results = approved_notes + rejected_scenarios
        current_date = datetime.now().strftime("%Y-%m-%d")
        sanitized_description = sanitize_filename(disease_description)
        results_df= pd.DataFrame(results)
        full_path= f"{path_to_save_notes}/{sanitized_description}_{current_date}.csv"
        results_df.to_csv(full_path, index=False, sep="|")


################ For ablation who epipeline with Llama 3.3 ends ############







################ For ablation whole pipeline with Qwen2.5-72B-Instruct begins ############

""" 
We already have these functions from previous ablation we could use:
        1. load_local_model

We need a few more functions for note generation:
        1. doctor_generate_scenario_with_llama
        2. doctor_generate_note_with_lamma
        3. polish_note_with_lamma
        4. generate_and_save_medical_notes_all_llama
        5. judge evaluate scneario biolm
"""


def judge_evaluate_scenario_with_qwen(scenario, judge_conversations_memory, model, tokenizer):
    """
    Evaluates a scenario using a custom model and returns the decision and response.
    
    Args:
        scenario (str): The scenario to evaluate.
        judge_conversations_memory (list): The conversation memory to maintain context.
        model: The model used for generation.
        tokenizer: The tokenizer used for encoding and decoding.
        generation_config (dict): Configuration for generation parameters.

    Returns:
        tuple: The decision (str) and the latest model response (str).
    """

    pre_prompt = [
        {"role": "system", "content": gen_constants.SCENARIO_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": scenario}
    ]
    prompt= pre_prompt + judge_conversations_memory

    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=gen_constants.scenario_judge_config["max_tokens"], #,
            temperature= 0.001, #gen_constants.scenario_judge_config["temperature"]: Cannot be 0 in tranformers
            top_p= gen_constants.scenario_judge_config["top_p"]
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    latest_message = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract the decision (e.g., the last word or a specific format)
    decision = latest_message.split()[-1]

    # Update the memory with the assistant's response
    judge_conversations_memory.append({"role": "assistant", "content": latest_message})

    # Debugging logs (optional)
    print(f"Latest message: {latest_message}")
    print(f"Decision: {decision}")
    print(f"Updated conversation memory length: {len(judge_conversations_memory)}")

    return decision, latest_message


def doctor_generate_scenario_with_qwen(condition, scenario_provider_memory, model, tokenizer):
    aci_train_df= pd.read_csv(gen_constants.ACI_TRAIN_SET_PATH)
    randome_index= random.randint(0, 66)
    aci_note_sample= aci_train_df["note"][randome_index]

    scenario_system_prompt= gen_constants.SCENARIO_PROVIDER_SYSTEM_PROMPT.format(EXAMPLE_NOTE=aci_note_sample)
    # Use the prompt template

    pre_prompt = [
        {"role": "system", "content": scenario_system_prompt},
        {"role": "user", "content": condition}
    ]
    prompt= pre_prompt + scenario_provider_memory

    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=gen_constants.scenario_generator_config["max_tokens"], #,
            temperature= gen_constants.scenario_generator_config["temperature"],
            top_p= gen_constants.scenario_generator_config["top_p"]
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    scenario = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"len of scenario_provider_memory is: {len(scenario_provider_memory)}")

    return scenario



def doctor_generate_note_with_qwen(scenario, model, tokenizer):
    aci_train_df= pd.read_csv(gen_constants.ACI_TRAIN_SET_PATH)
    randome_index= random.randint(0, 66)
    aci_note_sample= aci_train_df["note"][randome_index]
    note_generator_system_prompt= gen_constants.NOTE_GENERATOR_SYSTEM_PROMPT.format(EXAMPLE_NOTE=aci_note_sample)  

    pre_prompt = [
        {"role": "system", "content": note_generator_system_prompt},
        {"role": "user", "content": scenario}
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
            max_new_tokens=gen_constants.note_generator_config["max_tokens"], 
            temperature= gen_constants.note_generator_config["temperature"], 
            top_p= gen_constants.note_generator_config["top_p"]
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    note = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return note



def polish_note_with_qwen(note, model, tokenizer):
    note_polisher_system_prompt= gen_constants.NOTE_POLISHER_SYSTEM_PROMPT 
    
    pre_prompt = [
        {"role": "system", "content": note_polisher_system_prompt},
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
            max_new_tokens=gen_constants.note_polisher_config["max_tokens"], 
            temperature= 0.001, #gen_constants.note_generator_config["temperature"], 
            top_p= gen_constants.note_polisher_config["top_p"]
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    polished_note = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return polished_note


def abbreviate_note_all_qwen():
     return "ablation"


def generate_and_save_medical_notes_all_qwen(disease_description, notes_count, 
                                                    model, tokenizer,
                                                    path_to_save_notes):
    
    approved_notes = []
    rejected_scenarios = []
    judge_conversations_memory = []
    scenario_provider_memory= []
    try:
        while True:
            scenario = doctor_generate_scenario_with_qwen(disease_description, scenario_provider_memory, model, tokenizer)

            # if len(scenario_provider_memory) == 0:
            #     scenario_provider_memory.append({"role": "user", "content": disease_description})
                
            scenario_provider_memory.append({"role": "assistant", "content": scenario})

            decision, latest_message = judge_evaluate_scenario_with_qwen(scenario, judge_conversations_memory, model, tokenizer)
            print(decision)
            print(latest_message)
            if decision == "Go" or decision == "Go.":
                role = _extract_role(scenario)
                note = doctor_generate_note_with_qwen(scenario, model, tokenizer)
                polished_note = polish_note_with_qwen(note, model, tokenizer)
                abbreviated_note= abbreviate_note_all_qwen()
                approved_notes.append({"Disease Description": disease_description, "Scenario": scenario, "Note": note, "Polished Note": polished_note, "Abbreviated Note": abbreviated_note, "Role": role })
                scenario_provider_memory = []
                print(f"Note number {len(approved_notes)} has been generated!")
            else:
                rejected_scenarios.append({"Disease Description": disease_description, "Scenario": scenario, "Note": "Rejected", "Polished Note": "Rejected", "Abbreviated Note": "Rejected", "Role": "Rejected"})
                # to save on input tokens: drop the rejected scenario from the memory
                del judge_conversations_memory[-2:]

                scenario_provider_memory.append({"role": "user", "content": latest_message})

            # reset to respect the input token limit
            # the second condition prevents the occurance of infinite loops of rejecting scenarios.
            if  ((len(approved_notes) % 4) == 0) or (len(scenario_provider_memory) >= 6):
                     judge_conversations_memory = []

            # Respecting the number of needed approved notes
            if len(approved_notes) >= notes_count:
                break

    except Exception as e:
        print(f"An error occurred during note generation: {e}")
    
    finally:
        # Combine the results
        results = approved_notes + rejected_scenarios
        current_date = datetime.now().strftime("%Y-%m-%d")
        sanitized_description = sanitize_filename(disease_description)
        results_df= pd.DataFrame(results)
        full_path= f"{path_to_save_notes}/{sanitized_description}_{current_date}.csv"
        results_df.to_csv(full_path, index=False, sep="|")


################ For ablation who epipeline with Qwen2.5-72B-Instruct ends ############






def doctor_generate_note(scenario, openai_client):
        aci_train_df= pd.read_csv(gen_constants.ACI_TRAIN_SET_PATH)
        randome_index= random.randint(0, 66)
        aci_note_sample= aci_train_df["note"][randome_index]

        note_prompt= gen_constants.NOTE_GENERATOR_SYSTEM_PROMPT.format(EXAMPLE_NOTE=aci_note_sample)
        
        note_response = openai_client.chat.completions.create(
                model=gen_constants.note_generator_config["model"],
                temperature = gen_constants.note_generator_config["temperature"],
                max_tokens = gen_constants.note_generator_config["max_tokens"],
                top_p = gen_constants.note_generator_config["top_p"],
                frequency_penalty = gen_constants.note_generator_config["frequency_penalty"],
                presence_penalty = gen_constants.note_generator_config["presence_penalty"],
                messages=[
                    {
                        "role": "system",
                        "content": note_prompt #gen_constants.NOTE_GENERATOR_SYSTEM_PROMPT #doctor_note_generator_system_prompt
                    },
                    {
                        "role": "user",
                        "content": scenario
                    }
                ]
                )
        return note_response.choices[0].message.content






def polish_note(note, openai_client):
        note_response = openai_client.chat.completions.create(
                model=gen_constants.note_polisher_config["model"],
                temperature = gen_constants.note_polisher_config["temperature"],
                max_tokens = gen_constants.note_polisher_config["max_tokens"],
                top_p = gen_constants.note_polisher_config["top_p"],
                frequency_penalty = gen_constants.note_polisher_config["frequency_penalty"],
                presence_penalty = gen_constants.note_polisher_config["presence_penalty"],
                messages=[
                    {
                        "role": "system",
                        "content": gen_constants.NOTE_POLISHER_SYSTEM_PROMPT #note_polisher_system_prompt
                    },
                    {
                        "role": "user",
                        "content": note
                    }
                ]
                )
        return note_response.choices[0].message.content






def _extract_role(text):
    # take the first 5 lines
    first_5_lines = '\n'.join(text.splitlines()[:5])
    # Define the regular expression pattern to find "ROLE:" (in any capitalization) followed by any characters until a line break
    pattern = r"ROLE: (.+?)\n"

    match = re.search(pattern, first_5_lines, re.IGNORECASE)
    # If a match is found, return the group which matches the role description
    if match:
        return match.group(1)
    else:
        # Return None or an appropriate message if "ROLE:" is not found within the first 5 lines
        return "Role not found in the first 5 lines."
    


def sanitize_filename(filename):
    """Remove or replace characters that are not allowed in filenames."""
    return re.sub(r'[<>:"\\|?*/]', '-', filename)


# given a disease_description and number of required notes, generates notes.
def generate_and_save_medical_notes(disease_description, notes_count, openai_client,
                                    path_to_save_notes):
    
    approved_notes = []
    rejected_scenarios = []
    judge_conversations_memory = [{"role": "system", "content": gen_constants.SCENARIO_JUDGE_SYSTEM_PROMPT}]
    scenario_provider_memory= []
    try:
        while True:
            scenario = doctor_generate_scenario(disease_description, scenario_provider_memory, openai_client)

            # if len(scenario_provider_memory) == 0:
            #     scenario_provider_memory.append({"role": "user", "content": disease_description})
                
            scenario_provider_memory.append({"role": "assistant", "content": scenario})

            decision, latest_message = judge_evaluate_scenario(scenario, judge_conversations_memory, openai_client)
            if decision == "Go" or decision == "Go.":
                role = _extract_role(scenario)
                note = doctor_generate_note(scenario, openai_client)
                polished_note = polish_note(note, openai_client)
                abbreviated_note= abbreviate_note(polished_note, openai_client)
                approved_notes.append({"Disease Description": disease_description, "Scenario": scenario, "Note": note, "Polished Note": polished_note, "Abbreviated Note": abbreviated_note, "Role": role })
                scenario_provider_memory = []
                print(f"Note number {len(approved_notes)} has been generated!")
            else:
                rejected_scenarios.append({"Disease Description": disease_description, "Scenario": scenario, "Note": "Rejected", "Polished Note": "Rejected", "Abbreviated Note": "Rejected", "Role": "Rejected"})
                # to save on input tokens: drop the rejected scenario from the memory
                del judge_conversations_memory[-2:]

                scenario_provider_memory.append({"role": "user", "content": latest_message})

        
            # reset to respect the input token limit
            # the second condition prevents the occurance of infinite loops of rejecting scenarios.
            if  ((len(approved_notes) % 4) == 0) or (len(scenario_provider_memory) >= 6):
                     judge_conversations_memory = [{"role": "system", "content":gen_constants.SCENARIO_JUDGE_SYSTEM_PROMPT}]

            # Respecting the number of needed approved notes
            if len(approved_notes) >= notes_count:
                break

    except Exception as e:
        print(f"An error occurred during note generation: {e}")
    
    finally:
        # Combine the results
        results = approved_notes + rejected_scenarios
        current_date = datetime.now().strftime("%Y-%m-%d")
        sanitized_description = sanitize_filename(disease_description)
        results_df= pd.DataFrame(results)
        full_path= f"{path_to_save_notes}/{sanitized_description}_{current_date}.csv"
        results_df.to_csv(full_path, index=False, sep="|")


