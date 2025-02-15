import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfFolder

HF_WRITE_TOKEN = os.getenv('MY_HF_WRITE_TOKEN')
HF_READ_TOKEN = os.getenv('MY_HF_READ_TOKEN')

HfFolder.save_token(HF_WRITE_TOKEN)

from gen_utils import gen_constants

MODEL_PATH = f"/model-weights/Llama-3.3-70B-Instruct"

# Llama 3.3 uses the same template as Llama 3.1: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/
# Lamma 3.1 prompt template: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#prompt-template
# Also: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
user_query= """
"ROLE: Pulmonologist

PATIENT'S DEMOGRAPHIC INFORMATION
- Age: 68
- Gender: Male
- Ethnicity: Caucasian
- Socio-economic Status: Moderate
- Education Level: High school graduate
- Health Literacy: Moderate
- Job Status: Retired factory worker

CLINICAL SETTING
- Type: Pulmonology Specialty Clinic
- Geographical Location: Small City, Urban area, High pollution
- Neighborhood: Industrial area, Moderate access to healthcare facilities

TYPE OF ENCOUNTER
- Encounter: Follow-up

MEDICAL HISTORY
- Previous Diagnoses: Chronic Obstructive Pulmonary Disease (COPD) diagnosed 3 years ago
- Medication History: Currently on bronchodilator inhalers
- Allergies: None
- Chronic Conditions: Hypertension

SYMPTOM DESCRIPTION
- Severity: Moderate
- Duration: Persistent for the last 2 months
- Associated Symptoms: Increased shortness of breath, productive cough with yellow sputum
- Frequency: Daily episodes, worsened by exertion
- Impact on Daily Activities: Limited ability to perform household chores and walk short distances

PATIENT’S SELF-REPORTED HABITS AND LIFESTYLE
- Sleep: Difficulty in sleeping due to breathlessness
- Diet: High in processed foods
- Exercise: Minimal due to symptoms
- Tobacco Use: Smoked 1 pack per day for 40 years; quit 1 year ago
- Alcohol Consumption: Occasional social drinker
- Drug Use: Denies illicit drug use 
- Recreational Activities: Enjoys gardening but limited by symptoms

PATIENT'S BEHAVIOR
- Level of Cooperation: Compliant with medication but reluctant to use oxygen therapy suggested previously

PHYSICAL EXAMS
- Auscultation: Diminished breath sounds with wheezing and prolonged expiratory phase

INVESTIGATION/TEST RESULTS
- Awaiting: Chest X-ray to assess for changes in lung structure

MEDICAL OUTCOME
- Diagnosis: Chronic Obstructive Pulmonary Disease, Unspecified
- Prescribed Treatment: Inhaled corticosteroid (Fluticasone 250 mcg, 2 puffs twice daily), Continue bronchodilators
- Referral: Pulmonary Rehabilitation Program to improve exercise tolerance and breathing techniques
- Follow-up Recommendations: Review in 4 weeks to assess treatment efficacy and X-ray results
- Lifestyle Change: Encourage smoking cessation support programs, dietary improvements, and gradual increase in physical activity"
"""

prompt = f""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            {gen_constants.SCENARIO_JUDGE_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """



tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        local_files_only=True, 
        device_map="auto", 
    )

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# I'm not sure if presence_penalty and frequency_penalty are impelemented with the same logic as OpenAI, and also I used 0 for GPT model. So, I'm skipping them for Llama.
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=gen_constants.scenario_judge_config["max_tokens"], #,
        temperature= 0.001, #gen_constants.scenario_judge_config["temperature"]: Cannot be 0 in tranformers
        top_p= gen_constants.scenario_judge_config["top_p"]
    )

print(tokenizer.batch_decode(outputs, skip_special_tokens=False))


def load_local_model(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            local_files_only=True, 
            device_map="auto", 
        )
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
    
    prompt= pre_prompt.join(conversation_history)
    

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






''' 
def get_model_responses(self):
        for idx, conversation in enumerate (self.test_dataset["dialogue"]):
            print(f"processing idx: {idx}")

            inputs = self.tokenizer(
            [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> \n\n {{{{ {self.summarizer_system_promt} }}}}<|eot_id|><|start_header_id|>user<|end_header_id|> \n\n {{{{ This is the conversation: {conversation} }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"], return_tensors = "pt").to("cuda")
            
            outputs= self.model.generate(**inputs, 
                                         max_new_tokens= self.generation_config["max_new_tokens"], 
                                         use_cache = self.generation_config["use_cache"],
                                         do_sample= self.generation_config["do_sample"],
                                         temperature= self.generation_config["temperature"],
                                         top_p= self.generation_config["top_p"],)
                                         #repetition_penalty= self.generation_config["repetition_penalty"])
            

            response= self.tokenizer.batch_decode(outputs, skip_special_tokens = False) #True
            start_index = response[0].rfind("<|start_header_id|>assistant<|end_header_id|>")+45
            #end_index = response[0].rfind("<|eot_id|>")

            # Extract the summary part
            summary = response[0][start_index:].strip()
            print(f"len of response is: {len(response[0])}")
            print(f"len of summary is: {len(summary)}")
            if len(summary)<10:
                print("###########################################")
                print("response is: ",response[0])
                print("summary is: ",summary)

                


            dial_summary_pairs[idx]= {"conversation": conversation, "summary": summary}

            #if idx == 0:
            print(f"summary is: {summary}")

        return dial_summary_pairs





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
'''






"""

# Above didnt work, I ran int memory error. 
# Now, I'm trying this: https://huggingface.co/docs/transformers/perf_infer_gpu_multi: Beware that we need to change the launch too in the slurm code

# Initialize distributed
rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group("nccl", device_id=device)

# Retrieve tensor parallel model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    tp_plan="auto",
)

# Prepare input tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, 
                                          local_files_only=True)

inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Distributed run
outputs = model(inputs)
print(outputs)
""" 


# If it didn't work, first, try simply more gpus, like 5. If didn't wotk, go to the next steps. 
# Then try quantized versions, like 8bit or 4bit.
# Look at this if the above didn't work:https://medium.com/@aleksej.gudkov/how-to-run-llama-405b-a-comprehensive-guide-caeda184e997
# Also this: https://github.com/huggingface/accelerate/blob/main/examples/inference/pippy/llama.py
# Also this: https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-405b-multinode-16k-sampling.ipynb
# Also this: https://github.com/huggingface/huggingface-llama-recipes/blob/main/local_inference/fp8-405B.ipynb

