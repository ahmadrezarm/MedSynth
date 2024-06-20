import os


TCAIREM_OPENAI_API_KEY=os.getenv('TCAIREM_OPENAI_API_KEY')

ACI_TRAIN_SET_PATH="/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/input/TaskC-TrainingSet.csv"


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



NOTE_ABBREVIATOR_SYSTEM_PROMPT= """Assume you are a very experienced physician and you are conducting research. 
The research project is to generate synthetic medical notes from doctor-patient conversations. The notes must be in this format:
    ** 1. Subjective: This section includes the patient's own description of their symptoms and complaints.
    ** 2. Objective: This section includes observations and data gathered by the physician, such as vital signs, physical examination findings, and test results.
    ** 3. Assessment: This section includes the physician's evaluation of the patient's condition, including a diagnosis or differential diagnosis.
    ** 4. Plan: This section includes the physician's recommendations for treatment, management, and follow-up. 
    
You will be given a note. You taks is to make the note more similar to real notes by adding acronyms.

Here is an example of replacement:
Input Version:
<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>
    **4. Plan:** 
    1. Medications: 
    - Tramadol 50mg, oral, twice daily for severe pain if needed. 
    2. Treatment: 
    - Initiate physical therapy focused on pain management and improving range of motion. 
    3. Investigations: 
    - Order MRI of the thoracic spine to assess the extent of osteophytes and any spinal stenosis. 
    4. Patient Education and Follow-Up: 
    - Discussed the importance of adherence to prescribed medication and physical therapy. 
    - Advised on the necessity of MRI for better diagnostic clarity and potential surgical planning. 
    - Encouraged maintaining blood sugar and blood pressure control through medication and lifestyle changes. 
    - Return visit in 2 weeks for reassessment and review of MRI results. 
    
    5. Referral: 
    - Referral to a Neurosurgeon, Dr. Karen Mitchell, for further evaluation and to discuss potential surgical options if conservative measures fail. 
<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>

Output Version:
<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>
    Plan:
    Rx'ed tramadol 50mg PO BID PRN
    PT for pain mgmt and ROM
    MRI t-spine ordered ?spinal stenosis
    referred neurosx Dr K Mitchell for ?surgical options
    patient educated
<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>

Try to use accronyms that are populare in medicine in the whole note, not only the "Plan" section. Also, try to 
preserve the length of the note to at least 70 percent of the original lenth.

You cannot add or remove any information from the note, you can just replace terms with acronyms.
"""

scenario_generator_config= {"model": "gpt-4o", # "gpt-4-1106-preview"
                            "temperature": 1,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


scenario_judge_config= {"model": "gpt-4o", # "gpt-4-1106-preview"
                            "temperature": 0,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


note_generator_config= {"model": "gpt-4o", # "gpt-4-1106-preview"
                            "temperature": 0.9,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


note_polisher_config= {"model": "gpt-4o", # "gpt-4-1106-preview"
                            "temperature": 0,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


note_abbreviator_config= {"model": "gpt-4o", # "gpt-4-1106-preview"
                            "temperature": 0.2,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}



PATH_TO_SAVE_NOTES= "/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/output/notes_onVector"


# for semantic similarity
embeding_model= "text-embedding-3-small" #source: https://openai.com/api/pricing/






DELETE= """"
 ** 1. Subjective

        This is the first heading of the SOAP note. Documentation under this heading comes from the “subjective” experiences, personal views or feelings of a patient or someone close to them. In the inpatient setting, interim information is included here. This section provides context for the Assessment and Plan.

        *** Chief Complaint (CC)

        The CC or presenting problem is reported by the patient. This can be a symptom, condition, previous diagnosis or another short statement that describes why the patient is presenting today. The CC is similar to the title of a paper, allowing the reader to get a sense of what the rest of the document will entail.

        Examples: chest pain, decreased appetite, shortness of breath.
        However, a patient may have multiple CC’s, and their first complaint may not be the most significant one. Thus, physicians should encourage patients to state all of their problems, while paying attention to detail to discover the most compelling problem. Identifying the main problem must occur to perform effective and efficient diagnosis.

        *** History of Present Illness (HPI)

        The HPI begins with a simple one line opening statement including the patient's age, sex and reason for the visit.

        Example: 47-year old female presenting with abdominal pain.
        This is the section where the patient can elaborate on their chief complaint. An acronym often used to organize the HPI is termed “OLDCARTS”:

        Onset: When did the CC begin?
        Location: Where is the CC located?
        Duration: How long has the CC been going on for?
        Characterization: How does the patient describe the CC?
        Alleviating and Aggravating factors: What makes the CC better? Worse?
        Radiation: Does the CC move or stay in one location?
        Temporal factor: Is the CC worse (or better) at a certain time of the day?
        Severity: Using a scale of 1 to 10, 1 being the least, 10 being the worst, how does the patient rate the CC?
        It is important for clinicians to focus on the quality and clarity of their patient's notes, rather than include excessive detail.

        *** History
        You do not need to foloow the format below strictly, but you can inspire from it:

        Medical history: Pertinent current or past medical conditions
        Surgical history: Try to include the year of the surgery and surgeon if possible.
        Family history: Include pertinent family history. Avoid documenting the medical history of every person in the patient's family.
        Social History: An acronym that may be used here is HEADSS which stands for Home and Environment; Education, Employment, Eating; Activities; Drugs; Sexuality; and Suicide/Depression.
        
        *** Review of Systems (ROS)

        This is a system based list of questions that help uncover symptoms not otherwise mentioned by the patient.
        You do not need to foloow the format below strictly, but you can inspire from it:

        General: Weight loss, decreased appetite
        Gastrointestinal: Abdominal pain, hematochezia
        Musculoskeletal: Toe pain, decreased right shoulder range of motion
        Current Medications, Allergies

        Current medications and allergies may be listed under the Subjective or Objective sections. However, it is important that with any medication documented, to include the medication name, dose, route, and how often. 
        Example: Motrin 600 mg orally every 4 to 6 hours for 5 days

    ** 2. Objective: 
        This section documents the objective data from the patient encounter. This includes:
        You do not need to foloow the format below strictly, but you can inspire from it:

        Vital signs
        Physical exam findings
        Laboratory data
        Imaging results
        Other diagnostic data
        Recognition and review of the documentation of other clinicians.
        A common mistake is distinguishing between symptoms and signs. Symptoms are the patient's subjective description and should be documented under the subjective heading, 
        while a sign is an objective finding related to the associated symptom reported by the patient. An example of this is a patient stating he has “stomach pain,” which is a symptom, 
        documented under the subjective heading. Versus “abdominal tenderness to palpation,” an objective sign documented under the objective heading.
        
    ** 3. Assessment: 
        This section documents the synthesis of “subjective” and “objective” evidence to arrive at a diagnosis. This is the assessment of the patient’s status through analysis of the problem, possible interaction of the problems, and changes in the status of the problems. Elements include the following.
        You do not need to foloow the format below strictly, but you can inspire from it:
        *** Problem
        List the problem list in order of importance. A problem is often known as a diagnosis. Explain the reasons for the diagnoses. This is where the decision-making process is explained in depth.
        *** Differential Diagnosis
        This is a list of the different possible diagnosis, from most to least likely, and the thought process behind this list. This is where the decision-making process is explained in depth. Included should be the possibility of other diagnoses that may harm the patient, but are less likely.
    ** 4. Plan:
        This section details the need for additional testing and consultation with other clinicians to address the patient's illnesses. It also addresses any additional steps being taken to treat the patient. This section helps future physicians understand what needs to be done next. For each problem:
        You do not need to foloow the format below strictly, but you can inspire from it:

        *** State which testing is needed and the rationale for choosing each test to resolve diagnostic ambiguities; ideally what the next step would be if positive or negative
        *** Therapy needed (medications)
        *** Specialist referral(s) or consults
        *** Patient education, counseling
A comprehensive SOAP note has to take into account all subjective and objective information, and accurately assess it to create the patient-specific assessment and plan.
Try to write the notes in complete scentiences, and only sometimes use bullet points. For example, 
do not always list the objective sections as:
        Vital signs
        Physical exam findings
        Laboratory data
        Imaging results
        Other diagnostic data
        Recognition and review of the documentation of other clinicians.

Try to write it in a paragraph and include all the information. You can sometimes use this lists, not always, to ensure diversity in the styles of the notes.
"""




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

"For example, the general logical order of the conversation is: first discussing symptoms, then discussing the medical history, followed by discussing testing and results, and finally discussing treatment options, conclusioin etc."
DELETE_2= " If you find this conversation to be incoherent, you can try dividing it into two separate coherent conversations."



dial_generator_config= {"model": "gpt-4o", # "gpt-4-1106-preview"
                            "temperature": 0.7,
                            "max_tokens": 4095,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}



dial_polisher_config= {"model": "gpt-4o", # "gpt-4-1106-preview"
                            "temperature": 0.5,
                            "max_tokens": 4095,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}