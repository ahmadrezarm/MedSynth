
import pandas as pd
from datetime import datetime

from prometheus_eval import PrometheusEval

from prometheus_eval.litellm import LiteLLM, AsyncLiteLLM



from prometheus_eval.prompts import RELATIVE_PROMPT
from prometheus_eval.vllm import VLLM

from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

from utils import constants



# Relative Grading: Outputs A or B
def get_preference_score(conversation_list, reference_list, model_A_response_list, model_B_response_list):
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")  # prometheus-eval/prometheus-bgb-8x7b-v2.0
    preference_judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT) 
    preferences = {}
    for idx, reference in enumerate(reference_list):
        instruction_with_conversation = constants.prometheus_preference_instruction .format(conversation=conversation_list[idx])
        data = {
            "instruction": instruction_with_conversation,
            "response_A": model_A_response_list[idx],
            "response_B": model_B_response_list[idx],
            "reference_answer": reference,
            "rubric": constants.prometheus_preference_rubric
        }

        feedback, score = preference_judge.single_relative_grade(**data)
        preferences[idx] = {"conversation": conversation_list[idx], "reference_note": reference, 
                            "model_A_note": model_A_response_list[idx], "model_B_note": model_B_response_list[idx], 
                            "feedback": feedback, "Preference": score}
        print(f"Prometheus has processed Prefernce Score for index {idx}")

    return preferences



# Relative Grading: Outputs A or B
def get_preference_score_GPT(conversation_list, reference_list, model_A_response_list, model_B_response_list):
    model = AsyncLiteLLM('gpt-4o', requests_per_minute=1) # GPT-4 API (async generation considering rate limit)
    preference_judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)
    preferences = {}
    for idx, reference in enumerate(reference_list):
        instruction_with_conversation = constants.prometheus_preference_instruction .format(conversation=conversation_list[idx])
        data = {
            "instruction": instruction_with_conversation,
            "response_A": model_A_response_list[idx],
            "response_B": model_B_response_list[idx],
            "reference_answer": reference,
            "rubric": constants.prometheus_preference_rubric
        }

        feedback, score = preference_judge.single_relative_grade(**data)
        preferences[idx] = {"conversation": conversation_list[idx], "reference_note": reference, 
                            "model_A_note": model_A_response_list[idx], "model_B_note": model_B_response_list[idx], 
                            "feedback": feedback, "Preference": score}
        print(f"Prometheus has processed Prefernce Score for index {idx}")

    return preferences





# Absolute Grading: Outputs score of 1 to 5
def get_absolute_score(conversations_list, reference_list, model_response_list):
    absolute_judge = PrometheusEval(model_id="prometheus-eval/prometheus-7b-v2.0", absolute_grade_template=ABSOLUTE_PROMPT)
    absolute_scores = {}
    for idx, reference in enumerate(reference_list):
        instruction_with_conversation = constants.prometheus_absolute_instruction.format(conversation=conversations_list[idx])
        
        absolute_score_rubric = SCORE_RUBRIC_TEMPLATE.format(**constants.prometheus_absolute_rubric_data)
        
        feedback, score = absolute_judge.single_absolute_grade(
            instruction=instruction_with_conversation,
            response=model_response_list[idx],
            rubric=absolute_score_rubric,
            reference_answer=reference
        )
        
        absolute_scores[idx] = {"conversation": conversations_list[idx], "reference_note": reference, 
                                "model_note": model_response_list[idx], "feedback": feedback, "Score": score}
        
        print(f"Prometheus has processed Absolute Score for index {idx}")

    return absolute_scores



def _save_prometheus_scores(prometheus_scores, model_A_name, model_B_name= None, base_name= constants.PROMETHEUS_RESULT_BASE_NAME, 
                                path= constants.PATH_TO_SAVE_EVAL_OUTPUT):
                
                df = pd.DataFrame.from_dict(prometheus_scores, orient='index')
                current_date= datetime.now().strftime("%Y-%m-%d")

                if model_B_name:
                        full_name = f"{base_name}_{model_A_name}_{model_B_name}_{current_date}"

                else: 
                       full_name = f"{base_name}_{model_A_name}_{current_date}" 

                full_path = f"{path}/{full_name}.csv"
                df.to_csv(full_path, index=False, sep= "|")

    

def get_and_save_prometheus_absolute_scores(model_name,
                                            dial_summary_pairs_path):

        #dial_summary_pairs_df = pd.DataFrame.from_dict(dial_summary_pairs, orient='index')
        dial_summary_pairs_df= pd.read_csv(dial_summary_pairs_path, sep="|")
        prometheus_absolute_scores = get_absolute_score(
                                                        conversations_list= dial_summary_pairs_df["conversation"], 
                                                        reference_list= pd.read_csv(constants.Aci_test_path)["note"], 
                                                        model_response_list= dial_summary_pairs_df["summary"])
        
        _save_prometheus_scores(prometheus_scores= prometheus_absolute_scores, 
                                model_A_name= model_name)
        
        


def get_and_save_prometheus_preference_scores(model_A_name, model_B_name, 
                                              model_A_dial_summary_pairs_path, 
                                              model_B_dial_summary_pairs_path,
                                              test_data_path):
        
        model_A_dial_summary_pairs_df= pd.read_csv(model_A_dial_summary_pairs_path, sep="|")
        model_B_dial_summary_pairs_df= pd.read_csv(model_B_dial_summary_pairs_path, sep="|")
        #model_A_dial_summary_pairs_df = pd.DataFrame.from_dict(model_A_dial_summary_pairs, orient='index')
        #model_B_dial_summary_pairs_df = pd.DataFrame.from_dict(model_B_dial_summary_pairs, orient='index')
        prometheus_preference_scores = get_preference_score(conversation_list= model_A_dial_summary_pairs_df["conversation"], 
                                                            reference_list= pd.read_csv(test_data_path)["note"],  #pd.read_csv(constants.Aci_test_path), #, sep="|"["Polished Note"]
                                                            model_A_response_list= model_A_dial_summary_pairs_df["summary"],
                                                            model_B_response_list= model_B_dial_summary_pairs_df["summary"])
        
        _save_prometheus_scores(model_A_name= model_A_name, 
                                model_B_name= model_B_name, 
                                prometheus_scores= prometheus_preference_scores)



############### GPT ###########################

def get_and_save_prometheus_preference_scores_GPT(model_A_name, model_B_name, 
                                              model_A_dial_summary_pairs_path, 
                                              model_B_dial_summary_pairs_path, 
                                              test_data_path):
        
        model_A_dial_summary_pairs_df= pd.read_csv(model_A_dial_summary_pairs_path, sep="|")
        model_B_dial_summary_pairs_df= pd.read_csv(model_B_dial_summary_pairs_path, sep="|")
        #model_A_dial_summary_pairs_df = pd.DataFrame.from_dict(model_A_dial_summary_pairs, orient='index')
        #model_B_dial_summary_pairs_df = pd.DataFrame.from_dict(model_B_dial_summary_pairs, orient='index')
        prometheus_preference_scores = get_preference_score_GPT(conversation_list= model_A_dial_summary_pairs_df["conversation"],  
                                                            reference_list= pd.read_csv(test_data_path)["note"],#pd.read_csv(constants.Aci_test_path)["note"], #
                                                            model_A_response_list= model_A_dial_summary_pairs_df["summary"],
                                                            model_B_response_list= model_B_dial_summary_pairs_df["summary"])
        
        _save_prometheus_scores(model_A_name= f"GPT_{model_A_name}", 
                                model_B_name= model_B_name, 
                                prometheus_scores= prometheus_preference_scores)


############### GPT END ###########################




########################### Note2Dial #########################

# Relative Grading: Outputs A or B
def Note2Dial_get_preference_score(note_list, reference_list, model_A_response_list, model_B_response_list):
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")  # prometheus-eval/prometheus-bgb-8x7b-v2.0
    preference_judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT) 
    preferences = {}
    for idx, reference in enumerate(reference_list):
        instruction_with_note = constants.prometheus_Note_2_Dial_preference_instruction .format(note=note_list[idx])
        data = {
            "instruction": instruction_with_note,
            "response_A": model_A_response_list[idx],
            "response_B": model_B_response_list[idx],
            "reference_answer": reference,
            "rubric": constants.prometheus_Note_2_Dial_preference_rubric
        }

        feedback, score = preference_judge.single_relative_grade(**data)
        preferences[idx] = {"note": note_list[idx], "reference_dial": reference, 
                            "model_A_conversation": model_A_response_list[idx], "model_B_conversation": model_B_response_list[idx], 
                            "feedback": feedback, "Preference": score}
        print(f"Prometheus has processed Prefernce Score for index {idx}")

    return preferences



def Note_2_Dial_get_and_save_prometheus_preference_scores(model_A_name, model_B_name, 
                                              model_A_dial_summary_pairs_path, 
                                              model_B_dial_summary_pairs_path,
                                              test_data_path):
        
        model_A_dial_summary_pairs_df= pd.read_csv(model_A_dial_summary_pairs_path, sep="|")
        model_B_dial_summary_pairs_df= pd.read_csv(model_B_dial_summary_pairs_path, sep="|")
        #model_A_dial_summary_pairs_df = pd.DataFrame.from_dict(model_A_dial_summary_pairs, orient='index')
        #model_B_dial_summary_pairs_df = pd.DataFrame.from_dict(model_B_dial_summary_pairs, orient='index')
        prometheus_preference_scores = Note2Dial_get_preference_score(note_list= model_A_dial_summary_pairs_df.iloc[:, 0], 
                                                            reference_list= pd.read_csv(test_data_path)["dialogue"],  #pd.read_csv(constants.Aci_test_path)["note"], #
                                                            model_A_response_list= model_A_dial_summary_pairs_df["dialogue"],
                                                            model_B_response_list= model_B_dial_summary_pairs_df["dialogue"])
        
        _Note2Dial_save_prometheus_scores(model_A_name= model_A_name, 
                                model_B_name= model_B_name, 
                                prometheus_scores= prometheus_preference_scores)




def _Note2Dial_save_prometheus_scores(prometheus_scores, model_A_name, model_B_name= None, base_name= constants.PROMETHEUS_Note_2_Dial_RESULT_BASE_NAME, 
                                path= constants.PATH_TO_SAVE_EVAL_OUTPUT):
                
                df = pd.DataFrame.from_dict(prometheus_scores, orient='index')
                current_date= datetime.now().strftime("%Y-%m-%d")

                if model_B_name:
                        full_name = f"{base_name}_{model_A_name}_{model_B_name}_{current_date}"

                else: 
                       full_name = f"{base_name}_{model_A_name}_{current_date}" 

                full_path = f"{path}/{full_name}.csv"
                df.to_csv(full_path, index=False, sep= "|")



def Note2Dial_get_and_save_prometheus_preference_scores_GPT(model_A_name, model_B_name, 
                                              model_A_dial_summary_pairs_path, 
                                              model_B_dial_summary_pairs_path,
                                              test_data_path):
        
        model_A_dial_summary_pairs_df= pd.read_csv(model_A_dial_summary_pairs_path, sep="|")
        model_B_dial_summary_pairs_df= pd.read_csv(model_B_dial_summary_pairs_path, sep="|")
        #model_A_dial_summary_pairs_df = pd.DataFrame.from_dict(model_A_dial_summary_pairs, orient='index')
        #model_B_dial_summary_pairs_df = pd.DataFrame.from_dict(model_B_dial_summary_pairs, orient='index')
        prometheus_preference_scores = Note2Dial_get_preference_score_GPT(note_list= model_A_dial_summary_pairs_df.iloc[:, 0], 
                                                            reference_list= pd.read_csv(test_data_path)["dialogue"],  #pd.read_csv(constants.Aci_test_path)["note"], #
                                                            model_A_response_list= model_A_dial_summary_pairs_df["dialogue"],
                                                            model_B_response_list= model_B_dial_summary_pairs_df["dialogue"])
        
        _Note2Dial_save_prometheus_scores(model_A_name= f"GPT_{model_A_name}", 
                                model_B_name= model_B_name, 
                                prometheus_scores= prometheus_preference_scores)






def Note2Dial_get_preference_score_GPT(note_list, reference_list, model_A_response_list, model_B_response_list):
    model = AsyncLiteLLM('gpt-4o', requests_per_minute=1) # GPT-4 API (async generation considering rate limit)
    preference_judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)
    preferences = {}
    for idx, reference in enumerate(reference_list):
        instruction_with_note = constants.prometheus_Note_2_Dial_preference_instruction .format(note=note_list[idx])
        data = {
            "instruction": instruction_with_note,
            "response_A": model_A_response_list[idx],
            "response_B": model_B_response_list[idx],
            "reference_answer": reference,
            "rubric": constants.prometheus_Note_2_Dial_preference_rubric
        }

        feedback, score = preference_judge.single_relative_grade(**data)
        preferences[idx] = {"note": note_list[idx], "reference_dial": reference, 
                            "model_A_conversation": model_A_response_list[idx], "model_B_conversation": model_B_response_list[idx], 
                            "feedback": feedback, "Preference": score}
        print(f"Prometheus has processed Prefernce Score for index {idx}")

    return preferences

########################### Note2Dial END #########################


