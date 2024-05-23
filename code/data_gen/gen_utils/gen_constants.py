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

Just output the revised note, not anything else."""



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



Here are two examples of note and the corresponding conversation:


# Example 1: 
    ## Note:

    """

    CHIEF COMPLAINT

    Right knee pain.

    MEDICAL HISTORY

    The patient has a history of diabetes. She has been doing pretty good with her diet. She states that she forgets to check her sugars quite a bit.

    REVIEW OF SYSTEMS

    Musculoskeletal: Reports right knee pain and swelling.

    PHYSICAL EXAM

    Respiratory
    - Auscultation of Lungs: Clear bilaterally.

    Cardiovascular
    No murmurs, gallops.

    Musculoskeletal
    - Examination of the right knee: Some swelling present.
    - Palpation: Some pain to palpation on the medial aspect of the right knee, and a little bit of pain on the lateral aspect of the right knee.
    - Range of Motion: Limited range of motion as well as pain on both flexion and extension of the knee.
    - Special Testing:
    McMurray's Test: Negative.

    ASSESSMENT AND PLAN

    1. Right knee pain.
    - Medical Reasoning: I am concerned about a torn MCL due to pain on ambulation and trouble with weightbearing, as well as the pop she heard.
    - Patient Education and Counseling: We discussed treatment options today including bracing, anti-inflammatories, and icing. - Medical Treatment: I am going to put her in a straight leg brace and I will prescribe some Mobic. She can start taking that as a pain reliever and to try to get some of the swelling down. I want her to ice her knee once an hour for about 15 minutes.
    - Additional Testing: I am also going to send her out for an MRI.

    2. Type 2 diabetes.
    - Medical Reasoning: The patient states that her type 2 diabetes are well-managed.
    - Medical Treatment: I am also going to get a refill on the metformin that she has been taking 500 mg.
    - Additional Testing: We are going to recheck her hemoglobin A1c.

    Patient Agreements: The patient understands and agrees with the recommended medical treatment plan.
    """



    ## Conversation:

    """

    [doctor] so sophia i see that you you hurt your knee tell me about what happened
    [patient] yeah i was jumping on my kid's trampoline and i could just slipped out from under me
    [doctor] my gosh one of those big trampolines in your back yard
    [patient] yeah a pretty big one
    [doctor] okay which knee was it
    [patient] my right knee
    [doctor] right knee okay and when did this happen
    [patient] about four days ago
    [doctor] great the weather was perfect this weekend so i'm glad you at least got outside sorry to hear you got hurt okay so your right knee did you did you feel it like pop or or snap or anything when you hurt it
    [patient] yeah i felt a little pop and then it swelled up really big afterward
    [doctor] okay did you try anything for the pain
    [patient] i took some ibuprofen and i put some ice on it
    [doctor] okay did that help
    [patient] a little bit but it's still really hard to get around
    [doctor] alright and have you have you been able to stand on it or does that hurt too much
    [patient] it hurts quite a bit to stand but i am able to put weight on it
    [doctor] okay alright and what part of the knee is it inside outside middle
    [patient] kind of that inside part of my kneecap
    [doctor] okay alright and okay so as long as you're here and then your primary care physician i'm looking through your chart and it looks like we're treating your diabetes so how you've been doing with your your diet overall are you are you keeping your sugars low
    [patient] it's going okay i i forget to check quite a bit though
    [doctor] sure
    [patient] on it
    [doctor] yeah i understand how has your diet been lately
    [patient] it's been pretty good
    [doctor] okay okay good good you know it's hard to stay away from the sugary foods sometimes i i enjoy ice cream regularly okay so let's do physical exam as long as you are here so i'm just gon na listen to your heart your heart sounds normal no murmurs or gallops listen to your lungs quick if you can take a deep breath lungs are clear that's good news let's take a look at that knee right knee looks like it definitely has some swelling i'm gon na do some maneuvers here does it hurt when i push you on the inside of the knee
    [patient] yeah that hurts
    [doctor] okay how about the outside
    [patient] a little bit but not as much
    [doctor] okay so some pain on palpation on the inside little bit of pain on the outside of the knee if i bend the knee back does that hurt
    [patient] yeah
    [doctor] how about when i extend it
    [patient] yeah that hurts
    [doctor] okay so little bit of limited range of motion as well as pain on both flexion and extension on the knee i'm gon na push on this a little bit looks like your mcmurray's test is negative just checking for a meniscus tear okay so let's talk a little bit about your plan what i am concerned about for your knee is it sounds like you have a torn or injured mcl i it's that inside tendon in your knee so i'm concerned about that since you're having trouble with weightbearing and you heard that pop so what i'm gon na do is i'm gon na put you in a straight leg brace and i'll prescribe some mobic you can start taking that as a a pain reliever and to try to get some of the swelling down i want you to ice your knee once an hour for about fifteen minutes but i'm also gon na send you out for an mri because we wan na make sure this is what happens see if there's any other damage to the knee does that sound good
    [patient] yeah that sounds great thank you
    [doctor] yeah and then for your diabetes as long as you're here it sounds like you're managing that pretty well but i do wan na get a recheck on your hemoglobin a1c and then i'm also i'm going to get a refill on the metformin that you have been taking five hundred milligrams so you can keep taking that as well so do you have any other questions for me
    [patient] no that's it thanks
    [doctor] alright well thank you hope that you feel better

    """

    
# Example 2:

    ## Note:
    """
    CHIEF COMPLAINT

    Left shoulder pain.

    HISTORY OF PRESENT ILLNESS

    Alan Mitchell is a pleasant 69-year-old male who presents to the clinic today for the evaluation of left shoulder pain. The onset of his pain began 3 weeks ago, without any improvement. He denies any specific injury; however, he has been renovating his basement and putting in a new ceiling. He does not recall hitting or falling onto the left shoulder. The patient states he is very active and has experienced left shoulder pain before that usually resolves with Tylenol.

    The patient reports significant pain with reaching, lifting, and overhead activities. The pain is constant. He states the pain is primarily located in the left shoulder and denies it radiates down into the left arm. The patient also reports difficulty sleeping secondary to the pain. He denies any numbness or tingling in his left arm or fingers. He has been taking Tylenol for pain, which provides partial relief. He initially iced his shoulder but has not iced it recently. The patient denies he has done any physical therapy.

    REVIEW OF SYSTEMS

    Musculoskeletal: Reports left shoulder pain. Neurological: Denies numbness or tingling.

    VITALS

    All vital signs are within the normal limits.

    PHYSICAL EXAM

    MSK: Examination of the left shoulder: Limited active and passive ROM. Tenderness over the greater tuberosity of the humerus. No tenderness at the sternoclavicular or AC joints. Good hand grip. Neurovascularly intact distally. Capillary refill is less than 3 seconds. Sensation is intact to light touch distally.

    RESULTS

    X-rays of the left shoulder were obtained and reviewed today. These are normal and reveal no fracture or bony abnormalities.

    ASSESSMENT

    Left shoulder pain, likely rotator cuff tendinopathy.

    PLAN

    After reviewing the patient's examination and radiographic findings today, I have had a lengthy discussion with him regarding his current symptoms. I have explained that his x-rays did not reveal any signs of a fracture. I have recommended that we obtain an MRI of the left shoulder to evaluate for possible rotator cuff tendinopathy. The patient was provided with a referral to formal physical therapy. He will engage in a 6-to-8-week course in order to strengthen his left shoulder. I have also advised him to take Tylenol as needed for pain. If his symptoms do not improve, we may consider a steroid injection to the left shoulder.

    INSTRUCTIONS

    The patient will follow up with me once the MRI results are available for review and further discussion.
    """


    ## Conversation:

    """
    alright you can go ahead
    [patient] hey alan i good to see you today so i looked here my appointment notes and i see that you're coming in you had some shoulder pain left shoulder pain for the last three weeks so
    [doctor] how you doing is it is it gotten any better
    [patient] yeah yeah i've been having a lot of pain of my shoulder for the last three weeks now and it's not getting better okay do you remember what you were doing when the pain first started
    [doctor] so i i was thinking that i i ca n't recall like falling on it injuring it getting hit
    [patient] hmmm
    [doctor] i have been doing a lot of work in my basement and i even i put in a new ceiling so i do n't know if it's from all that activity doing that but otherwise that's that's all i can think of
    [patient] okay so do you remember hitting it or anything like that
    [doctor] no nothing at all
    [patient] okay alright did you fall do you remember doing that
    [doctor] no
    [patient] okay hmmm so like a little mystery so have you had pain in that shoulder before
    [doctor] i mean i'm very active so i can get pains in my shoulders but it's nothing that sometime some tylenol can help
    [patient] okay and are you able to move the arm or is it kinda just stuck
    [doctor] i'm having a lot of pain like i can move it but you know when i try to reach for something lifting anything and even like i do n't even try to put my hands over my head because it causes so much pain
    [patient] alright so does that pain radiate anywhere or like where would you say it is in your shoulder
    [doctor] it actually it stays pretty much just right at the shoulder it does n't go down anywhere
    [patient] okay and the pain is it is it all the time or does it come and go
    [doctor] it's pretty much all the time anytime i put any pressure on it like when i'm trying to sleep it hurts even more so it's been affecting my sleep as well
    [patient] okay so i know you mentioned tylenol so this time i have n't taken anything for it
    [doctor] yeah i i do the tylenol which usually works for me and it does take the edge off but i still have pain okay did you try icing it at all
    [patient] i iced it initially but i have n't iced it at all recently
    [doctor] alright
    [patient] and so with your shoulder have you experienced any numbness in your arm or in your fingers
    [doctor] no numbness or tingling
    [patient] okay good so i'm gon na go ahead and do a quick physical exam and take a look at your your shoulder so i reviewed your your vitals everything looks good with that so touch here in your shoulder so your left shoulder exam you have limited active and passive range of motion so pressure here so that there is tenderness of the greater
    [doctor] okay
    [patient] tuberosity of the humerus let's see there is no tenderness at the sternoclavicular or acro
    [doctor] yeah
    [patient] acromioclavicular joints
    [doctor] yeah yeah
    [patient] and looks like you have good hand grip let me see so on the neurovascular exam of your left arm your capillary refill is less than three seconds and your sensation is is intact to light touch
    [doctor] yes thank you yep
    [patient] so you did get a we get we had to get a x-ray of your shoulder before you came in and so it's normal so that's really good so there is no fractures no bony abnormalities so let's talk a little bit about my assessment and plan for you so you you do have that left shoulder pain so your symptoms are
    [doctor] most likely due to a rotator cuff tendinopathy so this means that you injured tendon you have injured tendons and muscles that make up your shoulder and make up your shoulder muscles so what i'm gon na do is i'm gon na order an mri of your left shoulder
    [patient] and so we're gon na begin with that just to make sure nothing else is going on have you done physical therapy before
    [doctor] i have n't
    [patient] okay so what i'm gon na do i'm going to refer you to physical therapy for approximately six to eight weeks and so they can help you strengthen those muscles around your shoulder and that should definitely help with the pain during that time you can also continue to take tylenol i do n't think i need to prescribe anything else for the pain you said as it's working pretty good for you so if your symptoms do n't improve we can consider a steroid injection of your shoulder which should provide some relief but i think right now we can just go with the the pt and hopefully that works to alleviate your injury so do you have any questions about the plan
    [doctor] so like i said i'm really active do you think that this pain will ever go away
    [patient] yeah so many patients are very successful with rehab and so we'll start with that and see how you do most most of the time once we build up those muscles around that shoulder you know things things the pain alleviates itself and and and you will be good to go back to working on your basement and running and jogging and lifting weights all all the active things people do these days
    [doctor] okay alright thank you
    [patient] bye
    [doctor] okay bye
    '""
'''

dial_generator_config= {"model": "gpt-4o", # "gpt-4-1106-preview"
                            "temperature": 0.59,
                            "max_tokens": 2236,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0}