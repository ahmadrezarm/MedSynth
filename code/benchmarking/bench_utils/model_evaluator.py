# source 1: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 
# source 2: https://huggingface.co/blog/mlabonne/orpo-llama-3

import os
import sys
import pandas as pd
from datetime import datetime

from unsloth import FastLanguageModel
import openai

from bench_utils import constants 

module_path = '/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/eval/continous_eval/utils/'
if module_path not in sys.path:
    sys.path.append(module_path)

from automatic_metrics import MetricsComputer 


class ModelEvaluatorAutoMetrics:
    def __init__(self, 
                 model, 
                 test_dataset_path= constants.TEST_DATA_PATH,       
                 metrics_file_path= constants.PATH_FOR_METRICS_DF, 
                 summarizer_system_promt = constants.summarizer_system_prompt):   
        
        self.metrics_file_path = metrics_file_path
        self.summarizer_system_promt= summarizer_system_promt

        # Load the existing metrics DataFrame if it exists, otherwise create a new one
        if os.path.exists(self.metrics_file_path):
            self.results_df = pd.read_csv(self.metrics_file_path, sep="|")
        else:
            self.results_df = pd.DataFrame()  # Initialize an empty DataFrame

        #self.model= model
        self.test_dataset= pd.read_csv(test_dataset_path, sep= "|")
            



        # Loading the fine-tuned model and the tokenizer for inference
        if "gpt" not in model.lower():
            self.model, self.tokenizer=  FastLanguageModel.from_pretrained(model_name = model,
                                                                            max_seq_length = constants.tuning_config.get("model_config").get("max_seq_length"),
                                                                            dtype = constants.tuning_config.get("model_config").get("dtype"),
                                                                            load_in_4bit = constants.tuning_config.get("model_config").get("load_in_4bit"),)

            # Using FastLanguageModel for fast inference
            FastLanguageModel.for_inference(self.model)


    
    def _initialize_openai_client(self):
        try:
            client = openai.Client(api_key= "sk-proj-6jNq4RERBKwdpFStA2mFT3BlbkFJdCWfVwfvNqkiwaR48VAm") #gen_constants.TCAIREM_OPENAI_API_KEY
            print("OpenAI client initialized successfully!")
            return client
        except Exception as e:
            print(f"An error occurred in initializing OpenAI API: {e}")
            return None
        
        

    def get_lamma3_responses(self):

        dial_summary_pairs = {}
        for idx, conversation in enumerate (self.test_dataset["polished_dial"]):
            print(f"processing idx: {idx}")

            inputs = self.tokenizer(
            [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> \n\n {{{{ {self.summarizer_system_promt} }}}}<|eot_id|><|start_header_id|>user<|end_header_id|> \n\n {{{{ This is the conversation: {conversation} }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"], return_tensors = "pt").to("cuda")
            
            outputs= self.model.generate(**inputs, 
                                         max_new_tokens= constants.non_gpt_eval_gen_config["max_new_tokens"], 
                                         use_cache = constants.non_gpt_eval_gen_config["use_cache"],
                                         do_sample= constants.non_gpt_eval_gen_config["do_sample"],
                                         temperature= constants.non_gpt_eval_gen_config["temperature"],
                                         top_p= constants.non_gpt_eval_gen_config["top_p"],)
            

            response= self.tokenizer.batch_decode(outputs, skip_special_tokens = False) #True
            start_index = response[0].rfind("<|start_header_id|>assistant<|end_header_id|>")+45
            #end_index = response[0].rfind("<|eot_id|>")

            # Extract the summary part
            summary = response[0][start_index:].strip()
            print(f"len of response is: {len(response[0])}")
            print(f"len of summary is: {len(summary)}")
            #print(f"summary is: {summary}")
            dial_summary_pairs[idx]= {"conversation": conversation, "summary": summary}

        return dial_summary_pairs





    def get_mistral_responses(self):
        dial_summary_pairs = {}
        for idx, conversation in enumerate (self.test_dataset["polished_dial"]):
            print(f"processing idx: {idx}")

            inputs = self.tokenizer(
            [f"<s>[INST]#### Instruction: \n {self.summarizer_system_promt} \n\n #### Input: \n This is the conversation: \n {conversation} [/INST]"], return_tensors = "pt",  add_special_tokens=True).to("cuda")
            
            outputs= self.model.generate(**inputs, 
                                         max_new_tokens= constants.non_gpt_eval_gen_config["max_new_tokens"], 
                                         use_cache = constants.non_gpt_eval_gen_config["use_cache"],
                                         do_sample= constants.non_gpt_eval_gen_config["do_sample"],
                                         temperature= constants.non_gpt_eval_gen_config["temperature"],
                                         top_p= constants.non_gpt_eval_gen_config["top_p"],
                                         )            

            # Extract the presciption part
            response= self.tokenizer.batch_decode(outputs, skip_special_tokens = False) #True
            start_index = response[0].rfind("[/INST]")+23

            # Extract the presciption part
            summary = response[0][start_index:].strip()
            print(f"len of response is: {len(response[0])}")
            print(f"len of summary is: {len(summary)}")
            #print(summary)
            dial_summary_pairs[idx]= {"conversation": conversation, "summary": summary}

        return dial_summary_pairs



    
    def get_gpt_responses(self):
        openai_client= self._initialize_openai_client()
        dial_summary_pairs = {}
        for idx, conversation in enumerate (self.test_dataset["polished_dial"]):
            print(f"processing idx: {idx}")
            reponse = openai_client.chat.completions.create(
                model= constants.gpt_config["model"],                 
                temperature = constants.gpt_config["temperature"],           
                max_tokens = constants.gpt_config["max_tokens"],
                top_p = constants.gpt_config["top_p"],
                messages=[
                    {
                        "role": "system",
                        "content": self.summarizer_system_promt #note_polisher_system_prompt
                    },
                    {
                        "role": "user",
                        "content": conversation
                    }
                ]
                )
            
            summary= reponse.choices[0].message.content
            
            dial_summary_pairs[idx]= {"conversation": conversation, "summary": summary}
        
        return dial_summary_pairs

    

    def get_lamma3_1_responses(self):
        pass
 


    def get_automatic_eval_scores(self, dial_summary_pairs, model_name):
        dial_summary_pairs_df = pd.DataFrame.from_dict(dial_summary_pairs, orient='index')
        
        # Debugging: Print the DataFrame to inspect
        print("Dial Summary Pairs DataFrame:")
        print(dial_summary_pairs_df.head())

        # Check if the required column exists in the test dataset
        if "Polished Note" not in self.test_dataset.columns:
            raise KeyError("Column 'Polished Note' not found in the dataset")

        # Ensure columns are strings for proper comparison
        dial_summary_pairs_df["summary"] = dial_summary_pairs_df["summary"].astype(str)
        self.test_dataset["Polished Note"] = self.test_dataset["Polished Note"].astype(str)

        # Debugging: Print the test dataset to inspect
        print("Test Dataset:")
        print(self.test_dataset[["Polished Note"]].head())

        print(f"dial_summary_pair[df]: {dial_summary_pairs_df['summary']}")
        print(f"self.test_dataset['Polished Note']: {self.test_dataset['Polished Note']}")

        metrics_computer = MetricsComputer(prediction_list= dial_summary_pairs_df["summary"],
                                            gt_list= self.test_dataset["Polished Note"])
        
        print("metrics compiter initialized sucessfuly!")
        eval_metrics= {
            'Model': model_name,
            'BLEU': metrics_computer.compute_BLEU(),
            'ROUGE-1': metrics_computer.compute_ROUGE()['rouge1'],
            'ROUGE-2': metrics_computer.compute_ROUGE()['rouge2'],
            'ROUGE-L': metrics_computer.compute_ROUGE()['rougeL'],
            'ROUGE-LSum': metrics_computer.compute_ROUGE()['rougeLsum'],
            'BERTScore': metrics_computer.compute_BERTScore(),
            "METEOR": metrics_computer.compute_METEOR(),
            'EvaluationDate': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        print(eval_metrics)
        # Append the current model's metrics to the results DataFrame
        eval_metrics_df = pd.DataFrame([eval_metrics])
        self.results_df = pd.concat([self.results_df, eval_metrics_df], ignore_index=True)

        return eval_metrics


    def save_model_output_to_csv(self, dial_summary_pairs, 
                            model_name, path= constants.PATH_TO_SAVE_BENCHMARK_OUTPUT):
        
        df = pd.DataFrame.from_dict(dial_summary_pairs, orient='index')
        current_date= datetime.now().strftime("%Y-%m-%d")
        full_name = f"{model_name}_{current_date}.csv"
        full_path = f"{path}/{full_name}"
        df.to_csv(full_path, index=False, sep="|")



    def save_eval_metrics_to_csv(self):
        # Save the accumulated results DataFrame to the CSV file
        self.results_df.to_csv(self.metrics_file_path, index=False, sep="|")

    ''' 
    def save_eval_metrics_to_csv(self, eval_metrics, metrics_filename,
                                  path):

        metrics_df = pd.DataFrame([eval_metrics])
        current_date = datetime.now().strftime("%Y-%m-%d")
        metrics_name = f"{metrics_filename}_{current_date}.csv"
        full_metrics_path = f"{path}/auto_metics_{metrics_name}"
        metrics_df.to_csv(full_metrics_path, index=False, sep="|")

    ''' 