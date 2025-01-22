import openai
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
""" 
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

"""

def judge_evaluate_scenario_no_judge(scenario, judge_conversations_memory, openai_client):
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

def abbreviate_note_no_judge(note, openai_client):
     return "ablation"

################# For ablation No Judge end #####################




################ For ablation Llama 3.1 as Judge begins ############

def judge_evaluate_scenario_llama_70b_judge(scenario, judge_conversations_memory, openai_client):
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


################ For ablation Llama 3.1 as Judge ends ############



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
                                    path_to_save_notes= gen_constants.PATH_TO_SAVE_NOTES):
    
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


