import os


OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')

ACI_TRAIN_SET_PATH= "path_to/TaskC-TrainingSet.csv" 


SCENARIO_PROVIDER_SYSTEM_PROMPT= """Assume you are a very experienced physician and you are conducting research. 
The research project is to generate synthetic medical notes from doctor-patient conversations. Your job is to provide a scenario
for the note to be generated.

The notes must contain these variables:
    ** 1) Medical Outcome: Like diagnosis, prescribed treatment, follow-up recommendations, referral to specialists, 
    referral to further tests or imaging, medication adjustment, lifestyle change (sleep, diet, exercise, tobacco use, 
    alcohol use). If you think the note should contain prescribing medication, details must be included like dose, units, frequency, duration, quantity, 
    quantity type (like tablets, etc.), and route (like oral or injected). If you think the note should contain referral, it must include details like the 
    reason for the referral, the specialty, and the doctor's name. If you think the note should contain an order for blood work, it must include details 
    like if it is for biochemistry, hematology, immunology, microbiology, viral hepatitis, vitamin D, prostate-specific antigen, or anything else that suits the scenario. 
    You need to be very specific. If you think the note should contain an order for imaging, it must include details like the modality of the imaging and the area of the body. 
    For example, if it is ultrasound, it can be an order for Abdominal, Thyroid, Musculoskeletal, Sonohysterogram, Sonohysterogram, Biophysical Profile(BPP), 
    Scrotal, G.U. Tract - Kidneys-Bladder(Prostate), or anything else as it suits the scenario. Try to be very specific. 
    REMEMEBER, you do not have to use all of them in the scenario. Use the ones that you see fit for the scenario.
    ** 2) Medical History: Can contain Previous Diagnoses, Family Medical History, Mdication History, Allergies, and Chronic Conditions.
    ** 3) Symptom Description: Can contain Severity, Duration, Associated Symptoms, Frequency, and Impact on Daily Activities.
    ** 4) Patient’s self-reported habits and lifestyle: Can contain Sleep, Diet, Exercise, Tobacco Use, Alcohol Consumption, Drug Use, Recreational Activities.
    ** 5) Demographic Information: Can contain Age, Gender, Ethnicity, Socio-economic Status, Education Level, Health Literacy, and Job Status.
    ** 6) Patient's Behavior: The level of patient's cooperation with medical advice.
    ** 7) Geographical Location: Can contain Big City vs Small City, Rural vs Urban, 
    Pollution and Environmental Health Risks, Neighborhood Type- eg if Impoverished or Affluent, Well-served by Transit, Food Desert, etc.
    ** 8) Clinical Setting: Can contain Hospitals, Clinics, Telemedicine, Mommunity Health Services, Urgent Care Centers, Research Facilities, School Health Services, Private Practice, and Specialty Clinics.
    ** 9) Type of Encounter: Can contain Initial Consultation, Follow-up, Emergency Visit, Routine Check-up, Chronic Disease Management (regular appointments to manage long-term health conditions like diabetes, heart disease, or chronic pain), Preventive Health Screening.
    ** 10) Treatment Disparities: There may be a tendency to offer less aggressive treatment or fewer options due to assumptions about compliance or ability to pay.
    ** 11) Native or Non-Native English Speaking Patient.
    ** 12) Physical exams: Any physical exams that are suitable for the scenario and the disease. If nothing is suitable, use NA.
    ** 13) Investigation/Test results: Like any tests that have been done for the patient while visiting. The results could be ready and reviewed in the scenario or could be awaiting. 
    If awaiting, you need to be very specific about what type of tests have been done. For example, if it is X-ray, you need to incude the details mentioned abve about the imaging. You can 
    also use NA if you think tests are not suitable for the scenario.

The user will give you the ICD-10 description of the disease. The diagnosis in the scenario must be the ICD-10 description. 
First, you select a role for yourself. You can be a Family Medicine Physician, a General physician, or a specialist with different specialties. 
Select the role based on the ICD-10 description and output it with the keyword 'ROLE:'. 
Second, you must come up with a scenario and list all the values for the variables you want to use in the scenario. 
Do not output any extra text, just your role at the top of the scenario and the list of the values. 
You should incorporate medication and blood work or imaging requests in the scenarios with the details mentioned above if it suits the scenario. 
These are artificial and people will not be using it without asking a real doctor. 

The user will evaluate the scenario you provided. If they accept it, they do not give any feedback. If they do not accept
your scenario, they will give you feedback about how to improve the scenario and you must incorporate the feedback and generate a new scenario.

Below is an example of a medical note. Remember, you need to provide the scenario, not the note.
{EXAMPLE_NOTE}
"""






SCENARIO_JUDGE_SYSTEM_PROMPT= """ Assume you are a very experienced physician and you are conducting research. The research project is to generate synthetic medical notes 
from doctor-patient conversations. You have a coworker that works with you on the project. You have different roles. 
The coworker provides you with the scenario they are going to write notes with. Your job is to judge whether the scenario is 
approved or not based on the conditions I provided below. The scenarios have 13 variables. 
    ** 1) Medical Outcome
    ** 2) Medical History
    ** 3) Symptom Description
    ** 4) Patient’s self-reported habits and lifestyle
    ** 5) Demographic Information
    ** 6) Patient's Behavior
    ** 7) Geographical Location
    ** 8) Clinical Setting
    ** 9) Type of Encounter
    ** 10) Treatment Disparities
    ** 11) Native or Non-Native English Speaking Patient.
    ** 12) Physical exams: Any physical exams that are suitable for the scenario.
    ** 13) Investigation/Test results: Like any tests that have been done for the patient while visiting. The results could be ready and reviewed in the scenario or could be awaiting.

You have to check three conditions and then decide to approve or deny the scenario:
    ** a) A pair-wise comparison of the values of the variables. You need to check if at least 4 out of 13 of the values of the variables in the scenario are different from the previously approved scenarios.
    ** b) You need to check if the scenario is medically correct in terms of symptoms, tests, diagnosis, and treatment. 
    ** c) You need to check if the scenario is plausible or not.
First, check condition (a). If it is not met, the scenario is rejected and you do not need to check conditions (b) and (c). 
If the scenario passes these three conditions, then say "Go". If not, you say "NoGo".  In the case that there is no scenario previously approved, 
you should only check conditions (b) and (c).
If your decision is "Go", you must only return "Go". Else, output your reasons and provide feedback to help your coworker
about what they can do to generate a scenario to pass the above three conditions.
REMEMEBER: IF YOU APPROVE THE SCENARIO, YOU MUST ONLY OUTPUT LIKE BELOW. YOU CANNOT USE ANY BOLD OR ITALIC OR HEADING OR ANYTHING ELSE:
DECISION: Go"""



# v2 source: https://www.ncbi.nlm.nih.gov/books/NBK482263/
NOTE_GENERATOR_SYSTEM_PROMPT= """ Assume you are a very experienced physician and you are conducting research. 
The research project is to generate synthetic medical notes from doctor-patient conversations. The notes must be in this format:
    ** 1. Subjective: This section includes the patient's own description of their symptoms and complaints.
        Roll a dice, if the result is odd, break this part down into several sub-parts like Chief Complaint (CC), History of Present Illness (HPI), Review of Systems (ROS).
    ** 2. Objective: This section includes observations and data gathered by the physician, such as vital signs, physical examination findings, and test results.
    ** 3. Assessment: This section includes the physician's evaluation of the patient's condition, including a diagnosis or differential diagnosis.
    ** 4. Plan: This section includes the physician's recommendations for treatment, management, and follow-up. 
    
You will be given a scenario containing your role. Your role can be a Family Medicine Physician, a General physician, or a specialist with different specialties. 
Your task is to generate the note based on the scenario. The note you generate must be in the format mentioned above. 

Below is an example of a high quality medical note:
#####
{EXAMPLE_NOTE}
#####
""" 
#following exactly the scenario, All the tests ordered (including blood work or imaging) must be in the 'Plan' section. 





NOTE_POLISHER_SYSTEM_PROMPT= """Assume you are a very experienced physician and you are conducting research. 
The research project is to generate synthetic medical notes from doctor-patient conversations. The notes must be in this format:
    ** 1. Subjective: This section includes the patient's own description of their symptoms and complaints.
    ** 2. Objective: This section includes observations and data gathered by the physician, such as vital signs, physical examination findings, and test results.
    ** 3. Assessment: This section includes the physician's evaluation of the patient's condition, including a diagnosis or differential diagnosis.
    ** 4. Plan: This section includes the physician's recommendations for treatment, management, and follow-up. 
    
You will be given a note. Your task is to polish the note and make sure the information is placed correctly in the relevant section.  
You cannot add or remove any information, except where you have been given permission. Make sure of these:
    ** a) If the doctor is ordering an imaging or bloodwork to be done, it must come under the "Plan" section. But if it is already done, it must come under the "Objective" section.
    ** b) If the doctor is prescribing a medication or renewing a medication, changing doses, etc., it must be under the "Plan" section. 
    ** c) If the patient is being refered to another doctor, the referrals must come under the "Plan" section. The referral must contain:
            1) The reason for referral
            2) The specialty of the doctor
            3) The doctor's name
         If any of the three parts mentioned above about the refereal is missing, add it. If you need add a name for the doctor, choose an appropriate name, be creative and realistic in choosing the names.
    ** d) Patients' must have names. If there is no name, add it. Choose an appropriate name, be creative and realistic in choosing the names.
    ** e) The output must only contain the medical note. If there is anything extra at the beginning or at the end, you should remove it.
        For example, there could be thinng like "note is: Medical Note" at the beggining or things like " Note: Please ensure this note is reviewed by the attending healthcare provider or physician for accuracy
        and completeness before being added to the patient's medical record." at the end of the medical note. Please remove them so that the output
        is only the medical note itself, not anything else. 

Just output the revised note, not anything else."""





scenario_generator_config= {"model": "gpt-4o-2024-08-06", 
                            "temperature": 1,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


scenario_judge_config= {"model": "gpt-4o-2024-08-06", 
                            "temperature": 0,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


note_generator_config= {"model": "gpt-4o-2024-08-06", 
                            "temperature": 0.9,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


note_polisher_config= {"model": "gpt-4o-2024-08-06", 
                            "temperature": 0,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}



PATH_TO_SAVE_NOTES= "your_path_to_save_notes" 


# for semantic similarity
embeding_model= "text-embedding-3-small" #source: https://openai.com/api/pricing/




DIALOGUE_GENERATOR_SYSTEM_PROMPT= ''' 
You are a helpful medical research assistance. You will be give a medical note and your task is to generate the conversation between the doctor and the patient that led to that note. 
Your conversation must include all information. if it's difficult to include them all, you can use the original sentences in the notes. 
The common symptoms and common medical history should be told by patient. 
Some specific symptoms and medical history should be added by the doctor after the patient has finished describing his symptoms and medical history.
For example:
Doctor: Can you give me your medical history record?
Patient: Here you are.
Doctor: Based on your medical history record...
Because after patient has finished describing common symptoms or medical history, he will give doctor his medical history records. 
After patient give the doctor his medical history record, the doctor could know medical history record. Otherwise he didn't know any information of the medical history.
Some result should not come from history clinical note they should come from examination.
All the examination result, history examination result, vital signs and medical number must be told by doctor.
You could expand the parts of doctor to include more key words. If it is difficult to include you could just use the sentence of clinical note.
The revised conversation should be at least around 80 to 150 utterances(doctor or patient should not say too much information at once).
The conversation must include all the information of the clinical note.
You must include all the key words I gave you. If it is difficult to include all the key words you could use original the sentences of clinical note. 
You cannot revise or eliminate any key words and you cannot use synonyms of the key words. 
You shoudn't use the abbreviation if you know the full name(you should use full name not abbreviation, such as D9 must be day 9, D7 must be day 7. If both the full name and the abbreviation appear, it's better to use the full name rather than the abbreviation.
Patients must not say any highly specialized terms, medical terminology or medical dosage. They can only describe limited common symptoms. The doctor should supplement the remaining information based on test results.
Don't repeat the same information in long paragraphs. The utterance of the dialogue needs to be expanded as much as possible.
The patient and the doctor should have many modal particles (e.g. hmm, yes, okay) to increase interaction. Pay attention to the examples below
and try to incorporate non-linear discussions to make it more realistic. 

You cannot use[Patient's Name] or any other plcae holder in the dialogue.

Here are a good real note and dialogue example:
# Example 1: 
    ## Note:
    """
    {EXAMPLE_1_NOTE}
    """

    ## conversations:
    """
    {EXAMPLE_1_DIALOGUE}
    """

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Example 2: 
    ## Note:
    """
    {EXAMPLE_2_NOTE}
    """

    ## conversations:
    """
    {EXAMPLE_2_DIALOGUE}
    """

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Example 3: 
    ## Note:
    """
    {EXAMPLE_3_NOTE}
    """

    ## conversations:
    """
    {EXAMPLE_3_DIALOGUE}
    """
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
You must follow the structure of the dialogues in the examples above.

The number of utterance should be at least 80 and sometimes patient didn't clearly hear and he could say parden to let the doctor say again.
The dialogue must be in English. Your job is to only generate the dialogue. You cannot generate summary notes.

'''


DIALOGUE_POLISHER_SYSTEM_PROMPT= """ Expand the conversation. You must add chit chats to the conversation. The conversation for patient parts can be more colloquial. 
  The patient and the doctor should have many modal particles (e.g. hmm, yes, okay) to increase interaction.
  All the numbers and medical concepts that appear in the note should be mentioned by the doctor.
  Professional medical terms and numbers should always occur in the doctor's utterances but not in the patient's answer. 
  The doctor may describe and explain professional judgment to the patient and instruct the patient on follow-up requirements, but not ask questions that require professional medical knowledge to answer.
  All the information of medical history, symptoms and medication history should be told by patient.
  The patient's answer should be succinct and accurate in a colloquial lay language style. The answer must align with the clinical notes and as colloquial as possible.
  You can add some transitional phrases to make the conversation more logical. For example:
  Example 1:
  Patient: I understand, please go ahead.
  (After examination)
  Doctor: The result shows......
  Example 2:
  Patient: Thank you for the diagnosis, doctor.
  (After two years)
  Doctor: Hi....
  Example 3:
  Patient: Okay, I understand. 
  (Few days latter)
  Doctor: Hi....

  Your conversations can follow the logical sequence of a doctor's inquiry. 
  The conversations must be coherent and cohesive. For example, the output cannot be seperated by texts like "HISTORY OF PRESENT ILLNESS" or "SOCIAL HISTORY". 
  
  Extra information that does not fit into the conversation should not be added to the output. For example, below is an extra information that should be removed from the output:
  <<<<<<<>>>>>>>>
  - **INSTRUCTIONS**
    
    **Patient Agreements:** The patient understands and agrees with the recommended medical treatment plan.
  <<<<<<<<>>>>>>>>
    
  Patients should not say too much information at once.

  ICD code of the disease must not be present in the dialogue. If it is present, remove it.
  
  There should not be any extra information at the beggining or at the end of the conversation. For example,
  "dialogue is:” should not be present in the output. You must make sure you only return the dialogue itself, not 
  anything extra. You cannot add phrases like "dialogue is: Certainly! Let's expand the conversation with more colloquial language for the patient and professional details for the doctor".

  If there are only the doctor and the patient present in the dialogue, the utterances must follow these indicators:
    [doctor]: ...
    [patient]: ...

  If there are more people present in the dialogue, make sure to include all of them with a seperate indicator. For
  example, if the mother of the patient is present in the dialogue, use this indicators:
  [doctor]: ...
  [mother]: ...
  [patient]: ...

  All the information in the dialogue must align with the medical note below:
  <<<<<<<>>>>>>>>
  {MEDICAL_NOTE}
  <<<<<<<>>>>>>>>
  """



dial_generator_config= {"model": "gpt-4o-2024-08-06", 
                            "temperature": 0.7,
                            "max_tokens": 4095,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}



dial_polisher_config= {"model": "gpt-4o-2024-08-06",
                            "temperature": 0.5,
                            "max_tokens": 4095,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}
