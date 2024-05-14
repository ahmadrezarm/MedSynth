import json
import torch
from datasets import load_dataset
from huggingface_hub import notebook_login
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from utils import constants

# Logging into the Hugging Face Hub(with token)
from huggingface_hub import HfFolder




class ModelTuner:

    def __init__(self, TRAINING_DATA_PATH_HF, FINE_TUNED_MODEL_NAME, tuning_config, base_model):

        self.tuning_config = tuning_config

        # Update the tuning configuration with actual values provided
        self.tuning_config['model_config']['base_model'] = base_model
        self.tuning_config['model_config']['finetuned_model'] = FINE_TUNED_MODEL_NAME
        self.tuning_config['training_dataset']['name'] = TRAINING_DATA_PATH_HF

    
    def _login_to_huggingface(self):
        HfFolder.save_token(constants.HF_WRITE_TOKEN)



    def _load_model_and_tokenizer(self):

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.tuning_config.get("model_config").get("base_model"),
            max_seq_length = self.tuning_config.get("model_config").get("max_seq_length"),
            dtype = self.tuning_config.get("model_config").get("dtype"),
            load_in_4bit = self.tuning_config.get("model_config").get("load_in_4bit"),
        )

    
    
    def _prepare_model_for_peft(self):
        # Setup for QLoRA/LoRA peft of the base model
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = self.tuning_config.get("lora_config").get("r"),
            target_modules = self.tuning_config.get("lora_config").get("target_modules"),
            lora_alpha = self.tuning_config.get("lora_config").get("lora_alpha"),
            lora_dropout = self.tuning_config.get("lora_config").get("lora_dropout"),
            bias = self.tuning_config.get("lora_config").get("bias"),
            use_gradient_checkpointing = self.tuning_config.get("lora_config").get("use_gradient_checkpointing"),
            random_state = 42,
            use_rslora = self.tuning_config.get("lora_config").get("use_rslora"),
            use_dora = self.tuning_config.get("lora_config").get("use_dora"),
            loftq_config = self.tuning_config.get("lora_config").get("loftq_config"),
        )
    


    def _load_training_data(self):
        # Loading the training dataset
        self.dataset_train = load_dataset(self.tuning_config.get("training_dataset").get("name"), split = self.tuning_config.get("training_dataset").get("split"))
    

    def _prepare_trainer(self):

        self.trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset_train,
            dataset_text_field = self.tuning_config.get("training_dataset").get("input_field"),
            max_seq_length = self.tuning_config.get("model_config").get("max_seq_length"),
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = self.tuning_config.get("training_config").get("per_device_train_batch_size"),
                gradient_accumulation_steps = self.tuning_config.get("training_config").get("gradient_accumulation_steps"),
                warmup_steps = self.tuning_config.get("training_config").get("warmup_steps"),
                max_steps = self.tuning_config.get("training_config").get("max_steps"),
                num_train_epochs= self.tuning_config.get("training_config").get("num_train_epochs"),
                learning_rate = self.tuning_config.get("training_config").get("learning_rate"),
                fp16 = self.tuning_config.get("training_config").get("fp16"),
                bf16 = self.tuning_config.get("training_config").get("bf16"),
                logging_steps = self.tuning_config.get("training_config").get("logging_steps"),
                optim = self.tuning_config.get("training_config").get("optim"),
                weight_decay = self.tuning_config.get("training_config").get("weight_decay"),
                lr_scheduler_type = self.tuning_config.get("training_config").get("lr_scheduler_type"),
                seed = 42,
                output_dir = self.tuning_config.get("training_config").get("output_dir"),
            ),
        )

    

    def model_train_and_save(self):
        
        self._login_to_huggingface()
        self._load_model_and_tokenizer()
        self._prepare_model_for_peft()
        self._load_training_data()
        self._prepare_trainer()

        self.trainer.train()

        # saving the model to the hub:
        self.model.push_to_hub(self.tuning_config.get("model_config").get("finetuned_model"), tokenizer= self.tokenizer, private= True)





    

    

