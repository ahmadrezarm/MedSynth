import os
import torch

# hf access tokens:
HF_WRITE_TOKEN = os.getenv('MY_HF_WRITE_TOKEN')
HF_READ_TOKEN = os.getenv('MY_HF_READ_TOKEN')



PATH_FOR_METRICS_DF= '/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/benchmarking/results/combined/benchmark_metrics.csv'


TEST_DATA_PATH= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/benchmarking/dataset/test.csv" 
PATH_TO_SAVE_BENCHMARK_OUTPUT= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/benchmarking/results"

HF_WRITE_TOKEN = os.getenv('MY_HF_WRITE_TOKEN')
HF_READ_TOKEN = os.getenv('MY_HF_READ_TOKEN')

summarizer_system_prompt= """You are an assistant for medical professionals, specializing in summarizing their conversations with patients. Your role is to accurately and comprehensively summarize these conversations in the SOAP (Subjective, Objective, Assessment, Plan) format. Ensure that each summary is thorough and precise, reflecting all relevant details from the conversation to provide a reliable medical record."""


gpt_config= {"model": "gpt-4o", # "gpt-4-1106-preview"
            "temperature": 0.5,
            "max_tokens": 4000,
            "top_p": 1,
            }


non_gpt_eval_gen_config = {"max_new_tokens":4000,
                     "do_sample":True,
                     "temperature":0.5, #0.6
                     "top_p":1,
                     "use_cache": True,
                    }



#base_model= "unsloth/llama-3-8b-Instruct" # changed to be consistent with the training.
#"meta-llama/Meta-Llama-3-8B-Instruct"


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
        "num_train_epochs": 1, # The number of training epochs(0 if the maximum steps are defined)
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



