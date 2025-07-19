# source 1: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 
# source 2: https://huggingface.co/blog/mlabonne/orpo-llama-3

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from huggingface_hub import HfFolder
import pandas as pd
from datetime import datetime

from unsloth import FastLanguageModel

from utils import constants 
from utils.automatic_metrics import MetricsComputer


HfFolder.save_token(constants.HF_WRITE_TOKEN)



class ModelEvaluatorAutoMetrics:
    def __init__(self, 
                 model, 
                 augmentor_system_promt= constants.dial_augmentor_system_prompt, 
                 test_dataset_path= constants.Aci_test_path, 
                 generation_config= constants.model_evaluator_generation_config):
        
        self.augmentor_system_promt= augmentor_system_promt
        #self.model= model
        self.test_dataset= pd.read_csv(test_dataset_path)
        self.generation_config = generation_config

        # Loading the fine-tuned model and the tokenizer for inference
        self.model, self.tokenizer=  FastLanguageModel.from_pretrained(model_name = model,
                                                                        max_seq_length = constants.tuning_config.get("model_config").get("max_seq_length"),
                                                                        dtype = constants.tuning_config.get("model_config").get("dtype"),
                                                                        load_in_4bit = constants.tuning_config.get("model_config").get("load_in_4bit"),)

        # Using FastLanguageModel for fast inference
        FastLanguageModel.for_inference(self.model)




    def get_model_responses(self):

        dial_summary_pairs = {}
        for idx, note in enumerate (self.test_dataset["note"]):
            print(f"processing idx: {idx}")

            inputs = self.tokenizer(
            [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> \n\n {{{{ {self.augmentor_system_promt} }}}}<|eot_id|><|start_header_id|>user<|end_header_id|> \n\n {{{{ This is the note: {note} }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"], return_tensors = "pt").to("cuda")
            
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
            dial = response[0][start_index:].strip()
            print(f"len of response is: {len(response[0])}")
            print(f"len of dial is: {len(dial)}")
            if len(dial)<10:
                print("###########################################")
                print("response is: ",response[0])
                print("summary is: ",dial)

                


            dial_summary_pairs[idx]= {"note": note, "dialogue": dial}

            #if idx == 0:
            print(f"dialogue is: {dial}")

        return dial_summary_pairs



    def get_automatic_eval_scores(self, dial_summary_pairs):
        dial_summary_pairs_df = pd.DataFrame.from_dict(dial_summary_pairs, orient='index')
        metrics_computer = MetricsComputer(prediction_list= dial_summary_pairs_df["dialogue"],
                                            gt_list= self.test_dataset["dialogue"])
        

        return {
            'BLEU': metrics_computer.compute_BLEU(),
            'ROUGE-1': metrics_computer.compute_ROUGE()['rouge1'],
            'ROUGE-2': metrics_computer.compute_ROUGE()['rouge2'],
            'ROUGE-L': metrics_computer.compute_ROUGE()['rougeL'],
            'ROUGE-LSum': metrics_computer.compute_ROUGE()['rougeLsum'],
            'BERTScore': metrics_computer.compute_BERTScore(),
            "METEOR": metrics_computer.compute_METEOR(),
            

            }


    def save_model_output_to_csv(self, dial_summary_pairs, 
                            name, path= constants.PATH_TO_SAVE_EVAL_OUTPUT):
        
        df = pd.DataFrame.from_dict(dial_summary_pairs, orient='index')
        current_date= datetime.now().strftime("%Y-%m-%d")
        full_name = f"{name}_{current_date}.csv"
        full_path = f"{path}/{full_name}"
        df.to_csv(full_path, index=False, sep="|")



    def save_eval_metrics_to_csv(self, eval_metrics, metrics_filename,
                                  path= constants.PATH_TO_SAVE_EVAL_OUTPUT):

        metrics_df = pd.DataFrame([eval_metrics])
        current_date = datetime.now().strftime("%Y-%m-%d")
        metrics_name = f"{metrics_filename}_{current_date}.csv"
        full_metrics_path = f"{path}/auto_metics_{metrics_name}"
        metrics_df.to_csv(full_metrics_path, index=False, sep="|")


