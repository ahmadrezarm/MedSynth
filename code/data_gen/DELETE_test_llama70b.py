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

