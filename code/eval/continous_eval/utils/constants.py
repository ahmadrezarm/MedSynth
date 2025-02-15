import os
import torch

# hf access tokens:
HF_WRITE_TOKEN = os.getenv('MY_HF_WRITE_TOKEN')
HF_READ_TOKEN = os.getenv('MY_HF_READ_TOKEN')



# for model eval
summarizer_system_prompt= """You are an assistant for medical professionals, specializing in summarizing their conversations with patients. Your role is to accurately and comprehensively summarize these conversations in the SOAP (Subjective, Objective, Assessment, Plan) format. Ensure that each summary is thorough and precise, reflecting all relevant details from the conversation to provide a reliable medical record."""

dial_augmentor_system_prompt= """You are an assistant for medical professionals, specializing in generating synthetic doctor-patient dialogues based on medical notes. Your role is to create dialogues that realistically reflect potential interactions between doctors and patients. The notes will be given to you in the SOAP (Subjective, Objective, Assessment, Plan) format. Ensure that each dialogue is realistic and informative, encapsulating all relevant details from the medical notes to simulate an authentic conversation."""


base_model= "unsloth/llama-3-8b-Instruct" # changed to be consistent with the training.
#"meta-llama/Meta-Llama-3-8B-Instruct"

Aci_test_path = "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/input/clinicalnlp_taskC_test2.csv"
Aci_train_path = "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/input/TaskC-TrainingSet.csv"

# useful link: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
model_evaluator_generation_config = {"max_new_tokens":3000,
                     "do_sample":True,
                     "temperature":0.6, #0.6
                     "top_p":0.9,
                     "use_cache": True,
                     # added
                     #"num_beams": 2,
                     #"no_repeat_ngram_size": 5,
                     #"repetition_penalty": 1.2,
                     #"length_penalty": ,
                     #"exponential_decay_length_penalty": (1800, -0.2), #(tuple(int, float), optional) — This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated.
                    }# #

PATH_TO_SAVE_EVAL_OUTPUT= "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/eval_results/cont_eval/ablations"

# for prometheus:
prometheus_preference_instruction = """
Imagine you are a medical professional tasked with evaluating summary notes taken from doctor-patient conversations. You will be given the conversation and ground truth note.
Each summary must accurately capture the key details and nuances of the conversation and ground truth note, including symptoms described by the patient (Subjective), observable facts and findings from the doctor (Objective), the doctor's diagnosis or interpretation of the patient's condition (Assessment), and the proposed treatment or next steps (Plan).
Some information might not be present in the conversation that are present in the ground truth note. This is because sometimes doctors write things in the note directly by looking at patient records. Ensure to consider both conversation and ground truth note in the evaluation.
Here is the conversation:
#############################
{conversation}
#############################

Here is the ground truth note:
#############################
{ground_truth_note}
#############################
"""



 
OLD_prometheus_preference_rubric= """
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




prometheus_preference_rubric = """ 
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

DELETED_ITEMS= """ 
3. Redundancy:
    - Does the summary note present clinical information succinctly without unnecessary repetition or redundant details?
    - Does every element contribute meaningfully to clarity and precision?


 include sections for Subjective, Objective, Assessment, and Plan? Subjective section can include or be subsituted by Chief Complaint (CC), History of Present Illness (HPI), History, Review of Systems (ROS), and Current Medications And Allergies.
        Objective section can include or be substituted by Vital signs, Physical exam findings, Laboratory data, Imaging results, Other diagnostic data, and Recognition and review of the documentation of other clinicians.
"""



prometheus_absolute_instruction = """
You are a medical professional evaluating summary notes taken from doctor-patient conversations. 
These conversations are summarized in the SOAP (Subjective, Objective, Assessment, Plan) format. 
Each summary should capture essential details and nuances of the conversation comprehensively and accurately. Note that some information in the note can come from external sources like patient's medical history in the EHR.
Therefore, make sure to evaluate the summary note against the provided ground truth note as well. 
Your task is to evaluate each summary note to ensure it captures the key components of the conversation and the ground truth note, employs medical terminology correctly, and organizes the information clearly and accurately according to the SOAP format.
Here is the conversation:
#############################
{conversation}
#############################

Here is the ground truth note:
#############################
{gt_note}
#############################

"""


prometheus_absolute_rubric_data= {"Hallucination": {"criteria": "Does the summary note accurately and comprehensively reflect the doctor–patient dialogue and ground truth note, free from any hallucinated or fabricated details that could distort clinical interpretation?",
                                                    "score1_description": "The note contains numerous fabricated or unsupported details not present in the actual dialogue or ground truth, significantly distorting the clinical picture and potentially leading to harmful decisions.",
                                                    "score2_description": "The note includes several hallucinated details that are not supported by the dialogue or ground truth, undermining overall reliability despite containing some accurate information.",
                                                    "score3_description": "The note is largely faithful to the original dialogue and ground truth but contain a few isolated hallucinated details. While these minor fabrications do not fundamentally alter the overall clinical message, they may slightly skew understanding.",
                                                    "score4_description": "The note is almost entirely consistent with both the dialogue and ground truth, with only rare and negligible hallucinated details that have minimal impact on clinical accuracy.",
                                                    "score5_description": "The note perfectly reflect the doctor–patient dialogue and match the ground truth in every detail, with no fabricated or unsupported information."}, 

                                  "Critical Omissions": {"criteria": "Does the summary note accurately and comprehensively capture all essential medical facts from the doctor–patient dialogue and ground truth, ensuring no critical omissions that could compromise clinical decision-making?",
                                                         "score1_description": "The note omits numerous essential medical facts from the dialogue and ground truth, resulting in a severely incomplete clinical record that could lead to harmful misunderstandings.",
                                                         "score2_description": "The note is missing several key medical facts, undermining the overall reliability of the clinical documentation despite containing some accurate information. ",
                                                         "score3_description": "While the note captures most of the essential medical facts, there are a few omissions that might slightly compromise clinical understanding. ",
                                                         "score4_description": "The note includes nearly all vital medical facts, with only rare omissions that have minimal impact on clinical accuracy. ",
                                                         "score5_description": "The note is fully comprehensive, capturing every critical medical fact from the doctor–patient dialogue and the ground truth note without any omissions."},
                                  
                                  "Redundancy": {"criteria": "Does the summary note present the clinical information succinctly and without unnecessary repetition or redundant details, ensuring that every piece of content contributes meaningfully to clarity and precision?",
                                                         "score1_description": "The note is burdened with excessive redundancy, featuring repeated phrases and duplicated information that severely disrupts clarity and wastes valuable space. This overabundance of repetition makes it very difficult for the reader to extract key clinical details.",
                                                         "score2_description": "The note shows a high level of redundancy with frequent, unnecessary repetitions that do little to enhance understanding. While some repeated elements may serve to emphasize key points, the overall effect is a cumbersome and disorganized narrative that requires significant editing.",
                                                         "score3_description": "The note contains moderate redundancy; there are instances of repeated information that do not add new insights and could be streamlined. Although the core message remains clear, reducing these repetitions would improve conciseness and overall readability.",
                                                         "score4_description": "The note is largely concise, with minimal redundancy. Occasional repetition is present but is used judiciously for emphasis or clarity, without detracting from the overall efficiency and professionalism of the documentation.",
                                                         "score5_description": "The note is impeccably succinct and free of unnecessary redundancy. Every sentence contributes new, relevant information, resulting in a clear, efficient, and highly professional clinical record."},

                                  "Professional Tone": {"criteria": "Does the summary note maintain a consistently professional tone appropriate for expert use, employing precise, formal, and respectful language throughout the documentation?",
                                                         "score1_description": "The note exhibits an unprofessional tone, with language that is overly casual, potentially disrespectful, or even inappropriate for expert medical documentation. The wording undermines the authority and credibility expected in clinical settings, detracting from the overall quality of the note.",
                                                         "score2_description": "The note occasionally lapses into informal or imprecise language that is not fully aligned with expert standards. While some sections maintain professionalism, there are noticeable moments where the tone is too casual, necessitating revisions to achieve consistency.",
                                                         "score3_description": "The note generally maintains a professional tone appropriate for expert use, though there are a few minor instances of casual phrasing or slight imprecision. Overall, the tone is acceptable, but refining these minor lapses could further enhance its credibility.",
                                                         "score4_description": "The note consistently uses formal and respectful language, demonstrating a strong professional tone with only very minor deviations that do not detract from its overall authority. The note reflects the standards expected in clinical documentation with minimal need for improvement.",
                                                         "score5_description": "The note exemplifies an impeccable professional tone, employing precise, formal, and respectful language throughout. Every aspect of the note aligns with expert standards, reinforcing its credibility and leaving no room for any casual or inappropriate expressions."},

                                  "The logical structure of the note and sentences": {"criteria": "Does the summary note exhibit a clear and logical structure with well-organized sentences and coherent transitions that enhance comprehension and accurately convey the clinical narrative?",
                                                         "score1_description": "The note's sentences are disjointed and lack any coherent logical structure, making the narrative extremely difficult to follow. There is little to no organization, with abrupt transitions and poorly constructed sentences that severely hinder comprehension.",
                                                         "score2_description": "The logical structure is weak, with attempts at organization that are often undermined by inconsistent sentence construction and unclear transitions. Although some parts may be understood, the overall note suffers from significant disorganization that can confuse the reader.",
                                                         "score3_description": "The note generally follows a logical structure, with most sentences organized in a coherent manner and transitions that, while occasionally abrupt, do not overwhelmingly disrupt the flow. Minor issues in sentence construction or clarity suggest that some revisions could further enhance the note’s overall coherence.",
                                                         "score4_description": "The sentences are well-organized and follow a clear logical progression, with effective transitions that largely guide the reader through the note. Minor structural imperfections may be present, but they do not detract significantly from the overall clarity and flow of the document.",
                                                         "score5_description": "The note demonstrates an exemplary logical structure, with every sentence and transition carefully crafted to create a seamless and coherent narrative. The organization is impeccable, enhancing the reader’s understanding and making the document exceptionally clear and professional."},

                                  "Adherence to SOAP format": {"criteria": "Does the summary note adhere to the SOAP format by delineating distinct and complete sections for Subjective, Objective, Assessment, and Plan?",
                                                         "score1_description": "The note completely disregards the SOAP format by omitting one or more of the required sections (Subjective, Objective, Assessment, and Plan), resulting in a chaotic and incomplete clinical record that severely hinders interpretation and decision-making.",
                                                         "score2_description": "The note attempts to adhere to the SOAP format but are missing at least one key section or improperly merge sections, leading to significant disorganization that undermines clarity and necessitates substantial revisions.",
                                                         "score3_description": "The note includes all four SOAP sections; however, there are occasional overlaps or blurred boundaries between sections. While the overall structure is present, these minor inconsistencies require correction to ensure clear and effective documentation.",
                                                         "score4_description": "The notes largely follow the SOAP format with each section clearly identifiable and most content appropriately placed. Minor deviations exist but do not notably detract from the overall clarity or clinical usefulness of the documentation.",
                                                         "score5_description": "The note flawlessly adheres to the SOAP format, with distinct and complete Subjective, Objective, Assessment, and Plan sections. This perfect organization enhances the clarity and clinical utility of the record without any errors or ambiguities."},

                                  "Section Relevance": {"criteria": "Does the summary note accurately assign clinical information to the correct sections—ensuring patient-reported details appear in the Subjective, objective findings in the Objective, clinician insights in the Assessment, and treatment strategies in the Plan—to enhance clarity and relevance?",
                                                         "score1_description": "The note demonstrates a severe lack of section relevance by placing crucial patient-reported details and clinical observations in the wrong sections—for example, mixing patient symptoms with objective findings or assessment notes—thus producing a distorted and confusing clinical narrative.",
                                                         "score2_description": "The note frequently misassigns information, with patient-reported details sometimes appearing in the Objective section and clinical findings misplaced in the Subjective section. Although some content is correctly located, these misclassifications significantly impair the record's clarity and reliability.",
                                                         "score3_description": " The note generally places information in the appropriate sections, with most patient-reported details in the Subjective area and clinical observations in the Objective section. However, occasional misplacements are evident, causing minor confusion that could slightly reduce the overall interpretability of the clinical record.",
                                                         "score4_description": "The note shows strong section relevance, with nearly all patient narratives, clinical findings, assessments, and plans appearing in their respective sections. Any minor misassignments are rare and have little impact on the clarity and effectiveness of the documentation.",
                                                         "score5_description": "The note exhibits perfect section relevance, ensuring that all patient-reported information is exclusively documented in the Subjective section, objective findings in the Objective section, the clinician’s reasoning in the Assessment section, and treatment plans in the Plan section."},                                                     
                                    }



PROMETHEUS_ABSOLUTE_RESULT_BASE_NAME = "Absolute_prometheus_scores"






############## N2D begins ##############

prometheus_Note_2_Dial_preference_instruction = """
Imagine you are a medical professional tasked with evaluating simulated doctor-patient conversations generated from summary notes. 
These conversations should accurately reconstruct interactions based on the provided note.

Here is the medical note:
#############################
{note}
#############################
"""

prometheus_Note_2_Dial_preference_rubric= """
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


PROMETHEUS_Note_2_Dial_RESULT_BASE_NAME = "prometheus_scores_Note_2_Dial"







# for model fine-tuning:

# Defining the configuration for the base model, LoRA and training
tuning_config = {
    "hugging_face_username":"Ahmad0067",
    "model_config": {
        "base_model":"{BASE_MODEL}", # The base model
        "finetuned_model":"{FINE_TUNED_MODEL_NAME}", #"llama-3-8b-Instruct-aci-train", # The finetuned model
        "max_seq_length": 8192, # The maximum sequence length that the base model can handle.
        "dtype":torch.bfloat16 , # The data type: changed from float16
        "load_in_4bit": True, # Load the model in 4-bit 
    },
    "lora_config": {
      "r": 16, # The number of LoRA layers 8, 16, 32, 64
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], # The target modules
      "lora_alpha":16, # The alpha value for LoRA
      "lora_dropout":0, # The dropout value for LoRA
      "bias":"none", # The bias for LoRA
      "use_gradient_checkpointing":True, # Use gradient checkpointing
      "use_rslora":False, # Use RSLora
      "use_dora":False, # Use DoRa
      "loftq_config":None # The LoFTQ configuration
    },
    "training_dataset":{
        "name":"{TRAINING_DATA_PATH_HF}", #"Ahmad0067/llama3_TaskC-TrainingSet_SOAP_instruct_dataset", # The dataset name(huggingface/datasets)
        "split":"train", # The dataset split
        "input_field":"prompt", # The input field
    },
    "training_config": {
        "per_device_train_batch_size": 2, # The batch size
        "gradient_accumulation_steps": 4, # The gradient accumulation steps
        "warmup_steps": 5, # The warmup steps
        "max_steps":0, # The maximum steps (0 if the epochs are defined)
        "num_train_epochs": 4, # The number of training epochs(0 if the maximum steps are defined)
        "learning_rate": 2e-4, # The learning rate
        "fp16": not torch.cuda.is_bf16_supported(), # The fp16
        "bf16": torch.cuda.is_bf16_supported(), # The bf16
        "logging_steps": 1, # The logging steps
        "optim" :"adamw_8bit", # The optimizer
        "weight_decay" : 0.01,  # The weight decay
        "lr_scheduler_type": "linear", # The learning rate scheduler
        "seed" : 42, # The seed
        "output_dir" : "outputs", # The output directory
    }
}




# NoteChat 

NOTE_CHAT_HF_PATH= "akemiH/NoteChat"
NUM_NOTE_CHAT_SAMPLES= 500
PATH_TO_SAVE_NOTE_CHAT_SAMPLES= "/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/data/input/NoteChatSamples"
NOTE_CHAT_SAMPLE_BASE_NAME= "note_chat_sample"