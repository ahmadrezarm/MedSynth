import os

PATH_FOR_METRICS_DF= '/Users/ahmadrezaie/2_My_papers/Synthetic_Data_Gen/code/benchmarking/results/combined/metrics.csv'


TEST_DATA_PATH= "/Users/ahmadrezaie/2_My_papers/Synthetic_Data_Gen/code/benchmarking/dataset/test.csv"
PATH_TO_SAVE_BENCHMARK_OUTPUT= "/Users/ahmadrezaie/2_My_papers/Synthetic_Data_Gen/code/benchmarking/results"

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


