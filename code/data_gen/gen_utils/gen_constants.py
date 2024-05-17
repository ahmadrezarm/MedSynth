import os


TCAIREM_OPENAI_API_KEY=os.getenv('TCAIREM_OPENAI_API_KEY')

ACI_TRAIN_SET_PATH="/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/TaskC-TrainingSet.csv"


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
    ** 12) Physical exams: Any physical exams that are suitable for the scenario and the disease.
    ** 13) Investigation/Test results: Like any tests that have been done for the patient while visiting. The results could be ready and reviewed in the scenario or could be awaiting. 
    If awaiting, you need to be very specific about what type of tests have been done. For example, if it is X-ray, you need to incude the details mentioned abve about the imaging.

The user will give you the ICD-10 description of the disease. The diagnosis in the scenario must be the ICD-10 description. 
First, you select a role for yourself. You can be a Family Medicine Physician, a General physician, or a specialist with different specialties. 
Select the role based on the ICD-10 description and output it with the keyword 'ROLE:'. 
Second, you must come up with a scenario and list all the values for the variables you want to use in the scenario. 
Do not output any extra text, just your role at the top of the scenario and the list of the values. 
You should incorporate medication and blood work or imaging requests in the scenarios with the details mentioned above if it suits the scenario. 
These are artificial and people will not be using it without asking a real doctor. 
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
    ** a) A pair-wise comparison of the values of the variables. You need to check if at least 5 out of 13 of the values of the variables in the scenario are different from the previously approved scenarios.
    ** b) You need to check if the scenario is medically correct in terms of symptoms, tests, diagnosis, and treatment. 
    ** c) You need to check if the scenario is plausible or not.
First, check condition (a). If it is not met, the scenario is rejected and you do not need to check conditions (b) and (c). 
If the scenario passes these three conditions, then say "Go". If not, you say "NoGo".  In the case that there is no scenario previously approved, 
you should only check conditions (b) and (c).
"Go" or "NoGo" must be your only outputs. """



# v2 source: https://www.ncbi.nlm.nih.gov/books/NBK482263/
NOTE_GENERATOR_SYSTEM_PROMPT= """ Assume you are a very experienced physician and you are conducting research. 
The research project is to generate synthetic medical notes from doctor-patient conversations. The notes must be in this format:
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

        Medical history: Pertinent current or past medical conditions
        Surgical history: Try to include the year of the surgery and surgeon if possible.
        Family history: Include pertinent family history. Avoid documenting the medical history of every person in the patient's family.
        Social History: An acronym that may be used here is HEADSS which stands for Home and Environment; Education, Employment, Eating; Activities; Drugs; Sexuality; and Suicide/Depression.
        
        *** Review of Systems (ROS)

        This is a system based list of questions that help uncover symptoms not otherwise mentioned by the patient.

        General: Weight loss, decreased appetite
        Gastrointestinal: Abdominal pain, hematochezia
        Musculoskeletal: Toe pain, decreased right shoulder range of motion
        Current Medications, Allergies

        Current medications and allergies may be listed under the Subjective or Objective sections. However, it is important that with any medication documented, to include the medication name, dose, route, and how often. 
        Example: Motrin 600 mg orally every 4 to 6 hours for 5 days

    ** 2. Objective: 
        This section documents the objective data from the patient encounter. This includes:

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
        *** Problem
        List the problem list in order of importance. A problem is often known as a diagnosis.
        *** Differential Diagnosis
        This is a list of the different possible diagnosis, from most to least likely, and the thought process behind this list. This is where the decision-making process is explained in depth. Included should be the possibility of other diagnoses that may harm the patient, but are less likely.
        Example: Problem 1, Differential Diagnoses, Discussion, Plan for problem 1 (described in the plan below). Repeat for additional problems.
    ** 4. Plan:
        This section details the need for additional testing and consultation with other clinicians to address the patient's illnesses. It also addresses any additional steps being taken to treat the patient. This section helps future physicians understand what needs to be done next. For each problem:
        *** State which testing is needed and the rationale for choosing each test to resolve diagnostic ambiguities; ideally what the next step would be if positive or negative
        *** Therapy needed (medications)
        *** Specialist referral(s) or consults
        *** Patient education, counseling
A comprehensive SOAP note has to take into account all subjective and objective information, and accurately assess it to create the patient-specific assessment and plan.
You will be given a scenario containing your role. Your role can be a Family Medicine Physician, a General physician, or a specialist with different specialties. 
You must generate the note following the scenario. All the notes you generate must be in the format mentioned above. """ #following exactly the scenario, All the tests ordered (including blood work or imaging) must be in the 'Plan' section. 





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

Just output the revised note, not anything else."""



scenario_generator_config= {"model": "gpt-4-1106-preview",
                            "temperature": 1,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


scenario_judge_config= {"model": "gpt-4-1106-preview",
                            "temperature": 0,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


note_generator_config= {"model": "gpt-4-1106-preview",
                            "temperature": 0.7,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}


note_polisher_config= {"model": "gpt-4-1106-preview",
                            "temperature": 0,
                            "max_tokens": 4000,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}



PATH_TO_SAVE_NOTES= "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/output/notes_onVector"


# for semantic similarity
embeding_model= "text-embedding-3-small" #source: https://openai.com/api/pricing/