# for base model eval
summarizer_system_prompt = """You are a an assistant to medical doctors and help them summarize their conversations with patients.
                The doctor will give you the conversation and you should summarize it into SOAP format. Please make sure it is comprehensive and accuarate. The summary will be used to
                as the medical note of the visit in Electronic Health Record system."""

base_model= "meta-llama/Meta-Llama-3-8B-Instruct"

Aci_test_path = "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/clinicalnlp_taskC_test2_SOAP.csv"
Aci_train_path = "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/TaskC-TrainingSet_SOAP.csv"


# for prometheus:
prometheus_preference_instruction = """
Imagine you are a medical professional tasked with evaluating summary notes taken from doctor-patient conversations. These conversations are summarized using the SOAP (Subjective, Objective, Assessment, Plan) format. Each summary must accurately capture the key details and nuances of the conversation, including symptoms described by the patient (Subjective), observable facts and findings from the doctor (Objective), the doctor's diagnosis or interpretation of the patient's condition (Assessment), and the proposed treatment or next steps (Plan).

You are to review each summary to ensure that it:

1. Accurately reflects the information provided during the conversation.
2. Is clearly organized according to the SOAP format.
3. Contains all relevant details needed for a comprehensive understanding of the patient’s situation.
4. Uses medical terminology correctly and appropriately.
5. Provides evidence-based assessments and plans where applicable.

#Here is the conversation:
#############################
#{conversation}
#############################

"""
# I ran out of gpu memory, removing conversations to see how it goes.

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
#Here is the conversation:

#############################
#{conversation}
#############################
"""

# I ran out of gpu memory, removing conversations to see how it goes.



prometheus_absolute_rubric_data = {
  "criteria":"Does the summary note accurately and comprehensively reflect the SOAP format with clarity and medical precision?",
  "score1_description":"The summary significantly lacks detail, has multiple inaccuracies, and fails to follow the SOAP format, making it potentially harmful or misleading in a clinical context.",
  "score2_description":"The summary includes basic elements of the SOAP format but omits important details or contains inaccuracies that could impede effective patient care. It shows a rudimentary use of medical terminology.",
  "score3_description":"The summary covers most necessary points and follows the SOAP format. There are minor inaccuracies or omissions that do not generally impede understanding or patient care. Medical terminology is used appropriately, with occasional errors.",
  "score4_description":"The summary is well-organized and follows the SOAP format closely, with only slight imperfections. It accurately captures the key components of the patient's condition and treatment plan. Medical terminology is used correctly and effectively.",
  "score5_description":"The summary excellently captures all aspects of the conversation accurately and comprehensively. It is perfectly aligned with the SOAP format, demonstrating professional-level use of medical terminology and a clear understanding of patient care."
}


