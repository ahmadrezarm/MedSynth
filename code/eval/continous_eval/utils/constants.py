import os
import torch

# hf access tokens:
HF_WRITE_TOKEN = os.getenv('MY_HF_WRITE_TOKEN')
HF_READ_TOKEN = os.getenv('MY_HF_READ_TOKEN')



# for model eval
summarizer_system_prompt= """You are an assistant for medical professionals, specializing in summarizing their conversations with patients. Your role is to accurately and comprehensively summarize these conversations in the SOAP (Subjective, Objective, Assessment, Plan) format. These summaries will serve as official medical notes for patient visits within the Electronic Health Record system. Ensure that each summary is thorough and precise, reflecting all relevant details from the conversation to provide a reliable medical record."""

base_model= "unsloth/llama-3-8b-Instruct-bnb-4bit" # changed to be consistent with the training.
#"meta-llama/Meta-Llama-3-8B-Instruct"

Aci_test_path = "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/clinicalnlp_taskC_test2_SOAP.csv"
Aci_train_path = "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/TaskC-TrainingSet_SOAP.csv"

# useful link: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
model_evaluator_generation_config = {"max_new_tokens":3000,
                     "do_sample":True,
                     "temperature":0.6, #0.6
                     "top_p":0.9,
                     "use_cache": True,
                     # added
                     #"num_beams": 2,
                     #"no_repeat_ngram_size": 5,
                     "repetition_penalty": 1.2,
                     #"length_penalty": ,
                     #"exponential_decay_length_penalty": (1800, -0.2), #(tuple(int, float), optional) — This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated.
                    }# #

PATH_TO_SAVE_EVAL_OUTPUT= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval"

# for prometheus:
prometheus_preference_instruction = """
Imagine you are a medical professional tasked with evaluating summary notes taken from doctor-patient conversations. These conversations are summarized using the SOAP (Subjective, Objective, Assessment, Plan) format. Each summary must accurately capture the key details and nuances of the conversation, including symptoms described by the patient (Subjective), observable facts and findings from the doctor (Objective), the doctor's diagnosis or interpretation of the patient's condition (Assessment), and the proposed treatment or next steps (Plan).
You are to review each summary to ensure that it:
1. Accurately reflects the information provided during the conversation.
2. Is clearly organized according to the SOAP format.
3. Contains all relevant details needed for a comprehensive understanding of the patient’s situation.
4. Uses medical terminology correctly and appropriately.
5. Provides evidence-based assessments and plans where applicable.
Here is the conversation:
#############################
{conversation}
#############################
"""


prometheus_preference_rubric= """
1. Completeness:
    - Does the summary include all significant components of the SOAP format?
    - Are there any crucial aspects of the conversation missing from the summary?
2. Accuracy:
    - How accurately does the summary reflect the details of the conversation as they were discussed?
    - Are there any discrepancies between what was said and what is noted?
3. Clarity and Structure:
    - Is the summary well-organized, following the logical flow of Subjective, Objective, Assessment, Plan?
    - Is the information presented in a clear and understandable manner?
4. Use of Medical Terminology:
    - Is medical terminology used correctly and effectively throughout the summary?
    - Does the use of terminology enhance the clarity and precision of the summary?
5. Evidence-Based Support:
    - In the Assessment and Plan sections, are the conclusions and recommendations supported by appropriate medical guidelines or literature?
    - Does the summary demonstrate a thoughtful and knowledgeable approach to patient care?
"""


prometheus_absolute_instruction = """
You are a medical professional evaluating summary notes taken from doctor-patient conversations. 
These conversations are summarized in the SOAP (Subjective, Objective, Assessment, Plan) format. 
Each summary should capture essential details and nuances of the conversation comprehensively and accurately.
Your task is to evaluate each summary note to ensure it captures the key components of the conversation, employs medical terminology correctly, and organizes the information clearly and accurately according to the SOAP format.
Here is the conversation:
#############################
{conversation}
#############################
"""


prometheus_absolute_rubric_data = {
  "criteria":"Does the summary note accurately and comprehensively reflect the SOAP format with clarity and medical precision?",
  "score1_description":"The summary significantly lacks detail, has multiple inaccuracies, and fails to follow the SOAP format, making it potentially harmful or misleading in a clinical context.",
  "score2_description":"The summary includes basic elements of the SOAP format but omits important details or contains inaccuracies that could impede effective patient care. It shows a rudimentary use of medical terminology.",
  "score3_description":"The summary covers most necessary points and follows the SOAP format. There are minor inaccuracies or omissions that do not generally impede understanding or patient care. Medical terminology is used appropriately, with occasional errors.",
  "score4_description":"The summary is well-organized and follows the SOAP format closely, with only slight imperfections. It accurately captures the key components of the patient's condition and treatment plan. Medical terminology is used correctly and effectively.",
  "score5_description":"The summary excellently captures all aspects of the conversation accurately and comprehensively. It is perfectly aligned with the SOAP format, demonstrating professional-level use of medical terminology and a clear understanding of patient care."
}


PROMETHEUS_RESULT_BASE_NAME = "prometheus_scores"



# for model fine-tuning:

# Defining the configuration for the base model, LoRA and training
tuning_config = {
    "hugging_face_username":"Ahmad0067",
    "model_config": {
        "base_model":"{BASE_MODEL}", # The base model
        "finetuned_model":"{FINE_TUNED_MODEL_NAME}", #"llama-3-8b-Instruct-aci-train", # The finetuned model
        "max_seq_length": 8192, # The maximum sequence length that the base model can handle.
        "dtype":torch.bfloat16 , # The data type: changed from float16
        "load_in_4bit": True, # Load the model in 4-bit 
    },
    "lora_config": {
      "r": 16, # The number of LoRA layers 8, 16, 32, 64
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], # The target modules
      "lora_alpha":16, # The alpha value for LoRA
      "lora_dropout":0, # The dropout value for LoRA
      "bias":"none", # The bias for LoRA
      "use_gradient_checkpointing":True, # Use gradient checkpointing
      "use_rslora":False, # Use RSLora
      "use_dora":False, # Use DoRa
      "loftq_config":None # The LoFTQ configuration
    },
    "training_dataset":{
        "name":"{TRAINING_DATA_PATH_HF}", #"Ahmad0067/llama3_TaskC-TrainingSet_SOAP_instruct_dataset", # The dataset name(huggingface/datasets)
        "split":"train", # The dataset split
        "input_field":"prompt", # The input field
    },
    "training_config": {
        "per_device_train_batch_size": 2, # The batch size
        "gradient_accumulation_steps": 4, # The gradient accumulation steps
        "warmup_steps": 5, # The warmup steps
        "max_steps":0, # The maximum steps (0 if the epochs are defined)
        "num_train_epochs": 4, # The number of training epochs(0 if the maximum steps are defined)
        "learning_rate": 2e-4, # The learning rate
        "fp16": not torch.cuda.is_bf16_supported(), # The fp16
        "bf16": torch.cuda.is_bf16_supported(), # The bf16
        "logging_steps": 1, # The logging steps
        "optim" :"adamw_8bit", # The optimizer
        "weight_decay" : 0.01,  # The weight decay
        "lr_scheduler_type": "linear", # The learning rate scheduler
        "seed" : 42, # The seed
        "output_dir" : "outputs", # The output directory
    }
}




# NoteChat 

NOTE_CHAT_HF_PATH= "akemiH/NoteChat"
NUM_NOTE_CHAT_SAMPLES= 50
PATH_TO_SAVE_NOTE_CHAT_SAMPLES= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/NoteChatSamples"
NOTE_CHAT_SAMPLE_BASE_NAME= "note_chat_sample"