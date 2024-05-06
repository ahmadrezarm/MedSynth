# Relative Grading: Outputs A or B

from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import RELATIVE_PROMPT

from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

from utils import constants



"""
def get_preference_score(conversations_list, refrence_list, model_A_response_list, model_B_response_list):
    preferences = {}
    for idx, refrence in enumerate(refrence_list):
        data = {
        "instruction": constants.prometheus_preference_instruction,
        "response_A": model_A_response_list[idx],
        "response_B": model_B_response_list[idx],
        "reference_answer": refrence,
        "rubric": constants.prometheus_preference_rubric
        }

        feedback, score = preference_judge.single_relative_grade(**data)
        preferences[idx] = {"feedback": feedback, "Preference": score}

    return preferences
"""

def get_preference_score(conversation_list, reference_list, model_A_response_list, model_B_response_list):
    preference_judge = PrometheusEval(model_id="prometheus-eval/prometheus-7b-v2.0", relative_grade_template=RELATIVE_PROMPT)
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
        print(f"index {idx} processed")

    return preferences



# Absolute Grading: Outputs score of 1 to 5
"""
def get_absolute_score(conversations_list, refrence_list, model_response_list):
    absolute_score_rubric = SCORE_RUBRIC_TEMPLATE.format(**constants.prometheus_absolute_rubric_data)
    absolute_scores = {}
    for idx, refrence in enumerate(refrence_list):
        feedback, score = absolute_judge.single_absolute_grade(
            instruction=constants.prometheus_absolute_instruction,
            response=model_response_list[idx],
            rubric=absolute_score_rubric,
            reference_answer=refrence
        )
        absolute_scores[idx] = {"feedback": feedback, "Score": score}

    return absolute_scores
""" 

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
        
        print(f"index {idx} processed")

    return absolute_scores








