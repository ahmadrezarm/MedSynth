import openai
import re
import pandas as pd
from datetime import datetime

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
        client = openai.Client(api_key=gen_constants.TCAIREM_OPENAI_API_KEY)
        print("OpenAI client initialized successfully!")
        return client
    except Exception as e:
        print(f"An error occurred in initializing OpenAI API: {e}")
        return None
    



def doctor_generate_scenario(condition, openai_client):
        aci_note_sample = pd.read_csv(gen_constants.ACI_TRAIN_SET_PATH)['note'].sample(n=1)
        scenario_prompt= scenario_prompt = gen_constants.SCENARIO_PROVIDER_SYSTEM_PROMPT.format(EXAMPLE_NOTE=aci_note_sample)
        
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
        ]
        )
        return scenario_response.choices[0].message.content



def doctor_generate_note(scenario, openai_client):
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
                "content": gen_constants.NOTE_GENERATOR_SYSTEM_PROMPT #doctor_note_generator_system_prompt
            },
            {
                "role": "user",
                "content": scenario
            }
        ]
        )
        return note_response.choices[0].message.content




def judge_evaluate_scenario(scenario, conversations_memory, openai_client):
    # Add system prompt and user scenario to memory
        conversations_memory += [
        {"role": "user", "content": scenario}
        ]
        evaluation_response = openai_client.chat.completions.create(
                model=gen_constants.scenario_judge_config["model"],
                temperature = gen_constants.scenario_judge_config["temperature"],
                max_tokens = gen_constants.scenario_judge_config["max_tokens"],
                top_p = gen_constants.scenario_judge_config["top_p"],
                frequency_penalty = gen_constants.scenario_judge_config["frequency_penalty"],
                presence_penalty = gen_constants.scenario_judge_config["presence_penalty"],
                messages=conversations_memory
                )
        # Accessing the last message's content correctly
        latest_message = evaluation_response.choices[0].message.content
        print("Latest message is: ", latest_message)
        decision = latest_message.split()[-1]  # Extract the last word

        # Update memory with the model's latest response
        conversations_memory.append({"role": "assistant", "content": latest_message})
        print(decision)
        print(conversations_memory)
        print("len of conversation_momory is: ",len(conversations_memory))
        return decision


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
    



# given a disease_description and number of required notes, generates notes.
def generate_and_save_medical_notes(disease_description, notes_count, openai_client,
                                    path_to_save_notes= gen_constants.PATH_TO_SAVE_NOTES):
    
    approved_notes = []
    rejected_scenarios = []
    judge_conversations_memory = [{"role": "system", "content": gen_constants.SCENARIO_JUDGE_SYSTEM_PROMPT}]

    try:
        while True:
            scenario = doctor_generate_scenario(disease_description, openai_client)
            decision = judge_evaluate_scenario(scenario, judge_conversations_memory, openai_client)
            if decision == "Go" or decision == "Go.":
                role = _extract_role(scenario)
                note = doctor_generate_note(scenario, openai_client)
                polished_note = polish_note(note, openai_client)
                approved_notes.append({"Disease Description": disease_description, "Scenario": scenario, "Note": note, "Polished Note": polished_note , "Role": role })
                
                print(f"Note number {len(approved_notes)} has been generated!")
            else:
                rejected_scenarios.append({"Disease Description": disease_description, "Scenario": scenario, "Note": "Rejected", "Polished Note": "Rejected", "Role": "Rejected"})
                # to save on input tokens: drop the rejected ones from the memory
                #judge_conversations_memory.pop()
        
            # reset to respect the input token limit
            if  (len(approved_notes) % 4) == 0:
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
        results_df= pd.DataFrame(results)
        full_path= f"{path_to_save_notes}/{disease_description}_{current_date}_v4.csv"
        results_df.to_csv(full_path, index=False, sep="|")


