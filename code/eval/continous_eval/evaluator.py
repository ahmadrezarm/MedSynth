# source: https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client
# using InferenceClient with Llama 3 is not free, so we just load and use it without InferenceClient.

# source 1: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 
# source 2: https://huggingface.co/blog/mlabonne/orpo-llama-3

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from huggingface_hub import HfFolder
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
import pandas as pd

from utils import access_tokens, constants 

HfFolder.save_token(access_tokens.hf_write_token)
#TODO: make the token an enviroenment variable

Aci_test = pd.read_csv(constants.Aci_test_path)
Aci_train_set = pd.read_csv(constants.Aci_train_path)

generation_config = {"max_new_tokens":2000,
                     "do_sample":True,
                     "temperature":0.6,
                     "top_p":0.9}


class Evaluator:
    def __init__(self, summarizer_system_promt = constants.summarizer_system_prompt, 
                 gt_test_dataset = Aci_test,  base_model = constants.base_model, gt_train_dataset= Aci_train_set, 
                 data_to_eval= None):
        
        self.summarizer_system_promt = summarizer_system_promt
        self.base_model = base_model
        self.gt_test_dataset = gt_test_dataset
        self.gt_train_dataset = gt_train_dataset
        self.data_to_eval = data_to_eval
        self.generation_config = generation_config


    def _prepare_model_for_test(self, model):
        """
        quantizes the model and makes it ready to be tested
        """
        # Flash Attention
        #if torch.cuda.get_device_capability()[0] >= 8:
            #attn_implementation = "flash_attention_2"
            #torch_dtype = torch.bfloat16
        #else:
            #attn_implementation = "eager"
            #torch_dtype = torch.float16

        # QLoRA config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, #torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            device_map="auto",
            #attn_implementation=attn_implementation
        )

        return model, tokenizer


    def _prepare_messages_for_test(self, messages, tokenizer, model):
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            return input_ids, terminators
            

    def test_model_on_test_set(self):
        model, tokenizer = self._prepare_model_for_test(self.base_model)
        dial_summary_pairs = {}
        for idx, conversation in enumerate (self.gt_test_dataset["dialogue"]):
            print(f"processing idx: {idx}")
            messages = [
                {"role": "system", "content": self.summarizer_system_promt},
                {"role": "user", "content": f"""{conversation}"""},
            ]

            input_ids, terminators = self._prepare_messages_for_test(messages, tokenizer, model)

            outputs = model.generate(
                input_ids,
                max_new_tokens= self.generation_config["max_new_tokens"],
                eos_token_id= terminators,
                do_sample= self.generation_config["do_sample"],
                temperature= self.generation_config["temperature"],
                top_p= self.generation_config["top_p"],
            )

            response = outputs[0][input_ids.shape[-1]:]
            summary = tokenizer.decode(response, skip_special_tokens=True)

            dial_summary_pairs[idx] = {"conversation": conversation, "summary": summary}

        return dial_summary_pairs
        
    
    def save_results_to_csv(self, dial_summary_pairs, name, path = "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/eval_results/cont_eval"):
        df = pd.DataFrame.from_dict(dial_summary_pairs, orient='index')
        
        full_path = f"{path}/{name}.csv"
        df.to_csv(full_path, index=False)


def main():
    evaluator = Evaluator()
    dial_summary_pairs = evaluator.test_model_on_test_set()
    evaluator.save_results_to_csv(dial_summary_pairs = dial_summary_pairs, name = "base_model")


if __name__ == '__main__':
    main()