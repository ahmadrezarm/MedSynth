import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from huggingface_hub import HfFolder
import os
import pandas as pd
import numpy as np
import random
from datetime import datetime
from utils import constants


HF_WRITE_TOKEN = os.getenv('MY_HF_WRITE_TOKEN')
HF_READ_TOKEN = os.getenv('MY_HF_READ_TOKEN')

HfFolder.save_token(HF_WRITE_TOKEN)



############ Dial-2-Note constants begins #####################

# prompts based on Prometheus: https://github.com/prometheus-eval/prometheus-eval/blob/62a85d118c497d17c1dddf1dbe61de1a77812fc9/eval/prompts.py#L1
dial_2_note_preference_instruction= """
Imagine you are a medical professional tasked with evaluating summary notes taken from doctor-patient conversations. You will be given the conversation and ground truth note.
Each summary must accurately capture the key details and nuances of the conversation and ground truth note, including symptoms described by the patient (Subjective), observable facts and findings from the doctor (Objective), the doctor's diagnosis or interpretation of the patient's condition (Assessment), and the proposed treatment or next steps (Plan).
Some information might not be present in the conversation that are present in the ground truth note. This is because sometimes doctors write things in the note directly by looking at patient records. Ensure to consider both conversation and ground truth note in the evaluation.

"""

dial_2_note_system_promt= """ 
You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.

###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{dial_2_note_preference_instruction}

"""




dial_2_note_preference_rubric= """ 
1. Hallucination:
    - Does the summary note accurately and comprehensively reflect the doctor–patient dialogue and ground truth note?

2. Critical Omissions:
    - Does the summary note capture all essential medical facts from the doctor–patient dialogue and ground truth note?

3. Professional Tone:
    - Does the summary note maintain a consistently professional tone appropriate for expert use?

4. Logical Structure:
    - Does the summary note exhibit a clear and logical structure?

5. Adherence to the Format:
    - Does the summary note follow the same structure as the ground-truth note?

6. Section Relevance:
    - Does the summary note accurately assign clinical information to the correct sections (e.g., patient-reported details in Subjective, objective findings in Objective, clinician insights in Assessment, and treatment strategies in Plan)?
"""

############ Dial-2-Note constants ends #####################


############ Note-2-Dial constants begins #####################
note_2_dial_preference_instruction= """
Imagine you are a medical professional tasked with evaluating simulated doctor-patient conversations generated from summary notes. 
These conversations should accurately reconstruct interactions based on the provided note.

"""

note_2_dial_system_promt= """ 
You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.

###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{note_2_dial_preference_instruction}

"""



note_2_dial_preference_rubric= """
1. Completeness:
    - Does the conversation cover all significant components of the note?
2. Accuracy:
    - How accurately does the conversation reflect the details of the note as they were recorded?
    - Are there any discrepancies between the note and the generated dialogue?
3. Naturalness and Flow:
    - Is the conversation realistic and natural, following a logical and smooth progression?
    - Does it sound like a genuine interaction between a doctor and patient?
4. Use of Medical Terminology:
    - Is medical terminology used correctly and effectively within the conversational context?
    - Does the use of terminology enhance the accuracy and professionalism of the conversation?
    - Does the technical terminology used by the doctor and the patient represent their respective knowledge levels (e.g., layperson language for the patient)?
5. Evidence-Based Support:
    - Are the doctor's statements and responses consistent with the medical details and recommendations in the notes?
"""


############ Note-2-Dial constants ends #####################



MODEL_PATH= "/model-weights/Qwen2.5-32B-Instruct"

def load_local_model(MODEL_PATH):

    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )
 
    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            quantization_config=quantization_config,
            device_map="auto", 
        )
    return model, tokenizer



def get_relative_score_qwen(relative_judge_model, relative_judge_tokenizer, conversation_list, reference_list, model_A_response_list, model_B_response_list):
    
    preferences = {}
    for idx, reference in enumerate(reference_list):
        user_prompt= """
        Here is the dialogue: 
        {conversation_list[idx]} 

        #####################

        Here is the ground truth note: 
        {reference}

        #####################

        Here is model A response:
        {model_A_response_list[idx]}

        #####################

        Here is model B response:
        {model_B_response_list[idx]}

        #####################

        Here is the Score Rubric:
        {dial_2_note_preference_rubric} 
        """
        prompt = [
        {"role": "system", "content": dial_2_note_system_promt.format(dial_2_note_preference_instruction= dial_2_note_preference_instruction)},
        {"role": "user", "content": user_prompt}]
        
        text = relative_judge_tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )

        model_inputs = relative_judge_tokenizer([text], return_tensors="pt").to(relative_judge_model.device)

        with torch.inference_mode():
            generated_ids = relative_judge_model.generate(
                **model_inputs,
                max_new_tokens=1000, #,
                temperature= 1,
                top_p= 0.9
            )


        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        feedback = relative_judge_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(feedback)

        preferences[idx] = {"conversation": conversation_list[idx], "reference_note": reference, 
                            "model_A_note": model_A_response_list[idx], "model_B_note": model_B_response_list[idx], 
                            "feedback": feedback,}

        print(f"Qwen has processed Prefernce Score for index {idx}")

    return preferences




def N2D_get_relative_score_qwen(relative_judge_model, relative_judge_tokenizer, note_list, reference_list, model_A_response_list, model_B_response_list,):
    
    preferences = {}
    for idx, reference in enumerate(reference_list):
        user_prompt= """
        Here is the note: 
        {note_list[idx]} 

        #####################

        Here is the ground truth dialogue: 
        {reference}

        #####################

        Here is model A response:
        {model_A_response_list[idx]}

        #####################

        Here is model B response:
        {model_B_response_list[idx]}

        #####################

        Here is the Score Rubric:
        {note_2_dial_preference_rubric} 
        """
        prompt = [
        {"role": "system", "content": note_2_dial_system_promt.format(note_2_dial_preference_instruction= note_2_dial_preference_instruction)},
        {"role": "user", "content": user_prompt}]
        
        text = relative_judge_tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )

        model_inputs = relative_judge_tokenizer([text], return_tensors="pt").to(relative_judge_model.device)

        with torch.inference_mode():
            generated_ids = relative_judge_model.generate(
                **model_inputs,
                max_new_tokens=1000, #,
                temperature= 1,
                top_p= 0.9
            )


        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        feedback = relative_judge_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(feedback)

        preferences[idx] = {"note": note_list[idx], "reference_dial": reference,
                            "model_A_conversation": model_A_response_list[idx], "model_B_conversation": model_B_response_list[idx],
                            "feedback": feedback,}

        print(f"Qwen has processed Prefernce Score for index {idx}")

    return preferences



def Note_2_Dial_get_and_save_qwen_preference_scores(model_A_name, model_B_name, 
                                              model_A_dial_summary_pairs_path, 
                                              model_B_dial_summary_pairs_path,
                                              test_data_path,
                                              relative_judge_model, 
                                              relative_judge_tokenizer,
                                              path_to_save):
        
        model_A_dial_summary_pairs_df= pd.read_csv(model_A_dial_summary_pairs_path, sep="|")
        model_B_dial_summary_pairs_df= pd.read_csv(model_B_dial_summary_pairs_path, sep="|")
        #model_A_dial_summary_pairs_df = pd.DataFrame.from_dict(model_A_dial_summary_pairs, orient='index')
        #model_B_dial_summary_pairs_df = pd.DataFrame.from_dict(model_B_dial_summary_pairs, orient='index')
        prometheus_preference_scores = N2D_get_relative_score_qwen(note_list= model_A_dial_summary_pairs_df.iloc[:, 0], 
                                                            reference_list= pd.read_csv(test_data_path)["dialogue"],  #pd.read_csv(constants.Aci_test_path)["note"], #
                                                            model_A_response_list= model_A_dial_summary_pairs_df["dialogue"],
                                                            model_B_response_list= model_B_dial_summary_pairs_df["dialogue"],
                                                            relative_judge_model= relative_judge_model,
                                                            relative_judge_tokenizer= relative_judge_tokenizer)
        
        _Note2Dial_save_qwen_scores(model_A_name= model_A_name, 
                                model_B_name= model_B_name, 
                                qwen_scores= prometheus_preference_scores, 
                                path= path_to_save)
        



def _Note2Dial_save_qwen_scores(qwen_scores, model_A_name, path, model_B_name= None, 
                                base_name= "N2D_Qwen_relative_score" ):
                
                df = pd.DataFrame.from_dict(qwen_scores, orient='index')
                current_date= datetime.now().strftime("%Y-%m-%d")

                if model_B_name:
                        full_name = f"{base_name}_{model_A_name}_{model_B_name}_{current_date}"

                else: 
                       full_name = f"{base_name}_{model_A_name}_{current_date}" 

                full_path = f"{path}/{full_name}.csv"
                df.to_csv(full_path, index=False, sep= "|")





def _save_qwen_relative_scores(prometheus_scores, model_A_name, base_name, model_B_name, path):
                
                df = pd.DataFrame.from_dict(prometheus_scores, orient='index')
                current_date= datetime.now().strftime("%Y-%m-%d")

                if model_B_name:
                        full_name = f"{base_name}_{model_A_name}_{model_B_name}_{current_date}"

                else: 
                       full_name = f"{base_name}_{model_A_name}_{current_date}" 

                full_path = f"{path}/{full_name}.csv"
                df.to_csv(full_path, index=False, sep= "|")


def get_and_save_qwen_preference_scores(model_A_name, model_B_name, 
                                              model_A_dial_summary_pairs_path, 
                                              model_B_dial_summary_pairs_path,
                                              test_data_path,
                                              path_to_save, 
                                              relative_judge_model, 
                                              relative_judge_tokenizer):
        
        model_A_dial_summary_pairs_df= pd.read_csv(model_A_dial_summary_pairs_path, sep="|")
        model_B_dial_summary_pairs_df= pd.read_csv(model_B_dial_summary_pairs_path, sep="|")
        #model_A_dial_summary_pairs_df = pd.DataFrame.from_dict(model_A_dial_summary_pairs, orient='index')
        #model_B_dial_summary_pairs_df = pd.DataFrame.from_dict(model_B_dial_summary_pairs, orient='index')
        prometheus_preference_scores = get_relative_score_qwen(conversation_list= model_A_dial_summary_pairs_df["conversation"], 
                                                            reference_list= pd.read_csv(test_data_path)["note"],  #pd.read_csv(constants.Aci_test_path), #, sep="|"["Polished Note"]
                                                            model_A_response_list= model_A_dial_summary_pairs_df["summary"],
                                                            model_B_response_list= model_B_dial_summary_pairs_df["summary"], 
                                                            relative_judge_model= relative_judge_model, 
                                                            relative_judge_tokenizer= relative_judge_tokenizer)
        
        _save_qwen_relative_scores(model_A_name= model_A_name, 
                                model_B_name= model_B_name, 
                                qwen_scores= prometheus_preference_scores,
                                path= path_to_save, base_name= "Qwen_relative_score")
        








comparisons_to_run= {
                     "pair00": {"model_A_name": "MS_10k_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/AhmadData_10k_samples_and_aci_train_model_2024-08-11.csv",
                               "model_B_name": "Base_model_No_Tune",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_base_model_no_tuning_2025-02-08.csv"},


                        "pair0": {"model_A_name": "Ab_NoJudge_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NoJudge_and_Aci_2025-02-08.csv",
                               "model_B_name": "AllGPT_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_Ahmad_246_Aci_2025-02-08.csv"},
                     
                     "pair1": {"model_A_name": "Ab_Ab_NoJudge_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NoJudge_and_Aci_2025-02-08.csv",
                               "model_B_name": "LammaJudgeGPT_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_LammaJudgeGPT_and_Aci_2025-02-08.csv"},
        
                            "pair2": {"model_A_name": "Ab_AllLamma57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_AllLamma_57Sample_and_Aci_2025-02-08.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Primock57_and_Aci_2024-12-14.csv"},

                            "pair3": {"model_A_name": "Ab_AllQwen57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/AllQwen_57_Samples_and_Aci_2025-02-08.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ACI_Primock57_and_Aci_2024-12-14.csv"},

                            "pair4": {"model_A_name": "Ab_AllLamma_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_AllLamma_and_Aci_2025-02-08.csv",
                               "model_B_name": "NC_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NC_246_Samples_and_Aci_2025-02-08.csv"},

                            "pair5": {"model_A_name": "Ab_AllQwen_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_AllQwen_and_Aci_2025-02-08.csv",
                               "model_B_name": "NC_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/ACI_NC_246_Samples_and_Aci_2025-02-08.csv"},
                  }


comparisons_to_run_remained= {
                     "pair1": {"model_A_name": "MS_10k_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/AhmadData_10k_samples_and_aci_train_model_2024-08-11.csv",
                               "model_B_name": "Aci_train_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/aci_train_model_2024-07-19.csv"},
}
''' 
def main():
        model, tokenizer= load_local_model(MODEL_PATH= MODEL_PATH)
        for pair_name, pair_data in comparisons_to_run_remained.items():
                print(f"Pair is {pair_name} and Model A name is: {pair_data['model_A_name']}")
                get_and_save_qwen_preference_scores(model_A_name= pair_data["model_A_name"],
                                                        model_B_name= pair_data["model_B_name"],
                                                        model_A_dial_summary_pairs_path= pair_data["model_A_data_path"],
                                                        model_B_dial_summary_pairs_path= pair_data["model_B_data_path"],
                                                        test_data_path=constants.Aci_test_path,
                                                        path_to_save= "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/llm_eval_v2/reltive_scores",
                                                        relative_judge_model= model,
                                                        relative_judge_tokenizer= tokenizer)
                



''' 

N2D_comparisons_to_run = {
                           "pair2": {"model_A_name": "N2D_MS_10k_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_Ahmad_and_Aci_2024-08-30.csv",
                               "model_B_name": "NC_10k_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_NoteChat_and_Aci_2024-08-30.csv"},

                           "pair3": {"model_A_name": "N2D_MS_10k_Only",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_Ahmad_Only_2024-08-30.csv",
                               "model_B_name": "NC_10k_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_NoteChat_Only_2024-08-30.csv"},


                           "pair4": {"model_A_name": "N2D_MS_57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Ahmad57_and_Aci_2024-12-14.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock57_and_Aci_2024-12-14.csv"},


                           "pair5": {"model_A_name": "N2D_MS_57_Only",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Ahmad57_Only_2024-12-14.csv",
                               "model_B_name": "PriMock_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock_Only_2024-12-09.csv"},


                           "pair6": {"model_A_name": "N2D_NoJudge_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_NoJudge_and_Aci_2025-02-08.csv",
                               "model_B_name": "AllGPT_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_Ahmad_246_Aci_2025-02-09.csv"},


                           "pair7": {"model_A_name": "N2D_NoJudge_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_NoJudge_and_Aci_2025-02-08.csv",
                               "model_B_name": "LammJudgeGPT_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_LammaJudgeGPT_and_Aci_2025-02-08.csv"},


                           "pair8": {"model_A_name": "N2D_AllLamma_57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_AllLamma_57Sample_and_Aci_2025-02-09.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock57_and_Aci_2024-12-14.csv"},


                           "pair9": {"model_A_name": "N2D_AllQwen_57_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_AllQwen_57_Samples_and_Aci_2025-02-09.csv",
                               "model_B_name": "PriMock_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/N2D_ACI_Primock57_and_Aci_2024-12-14.csv"},


                           "pair10": {"model_A_name": "N2D_AllLamma_246_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_AllLamma_and_Aci_2025-02-08.csv",
                               "model_B_name": "NC_246_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_NC_246_Samples_and_Aci_2025-02-09.csv"},


                           "pair11": {"model_A_name": "N2D_AllQwen_246_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_AllQwen_and_Aci_2025-02-09.csv",
                               "model_B_name": "NC_246_and_Aci",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations/N2D_ACI_NC_246_Samples_and_Aci_2025-02-09.csv"},


                            "pair12": {"model_A_name": "N2D_MS_10k_and_Aci",
                               "model_A_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_Ahmad_and_Aci_2024-08-30.csv",
                               "model_B_name": "Aci_train_Only",
                               "model_B_data_path": "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/Note2Dial_llama-3-8b-Instruct_Aci-Train_Only_2024-08-30.csv"},
                        }





def N2D_main():
        model, tokenizer= load_local_model(MODEL_PATH= MODEL_PATH)
        for pair_name, pair_data in N2D_comparisons_to_run.items():
                print(f"Pair is {pair_name} and Model A name is: {pair_data['model_A_name']}")
                Note_2_Dial_get_and_save_qwen_preference_scores(model_A_name= pair_data["model_A_name"],
                                                        model_B_name= pair_data["model_B_name"],
                                                        model_A_dial_summary_pairs_path= pair_data["model_A_data_path"],
                                                        model_B_dial_summary_pairs_path= pair_data["model_B_data_path"],
                                                        test_data_path=constants.Aci_test_path,
                                                        path_to_save= "//h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/llm_eval_v2/reltive_scores/N2D/qwen_fixed",
                                                        relative_judge_model= model,
                                                        relative_judge_tokenizer= tokenizer)


if __name__ == '__main__':
        N2D_main()
