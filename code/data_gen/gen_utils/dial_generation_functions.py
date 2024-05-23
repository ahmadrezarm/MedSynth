import openai
import re
import pandas as pd
from datetime import datetime
import random

import gen_constants #from . 

# make the openai key an env variable and load it here
def initialize_openai_client():
    """
    Initialize the OpenAI client with an API key read from a file.

    Args:
    api_key_file (str): The path to the file containing the OpenAI API key.

    Returns:
    openai.Client: An instance of the OpenAI client.
    """
    try:
        client = openai.Client(api_key= "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") #gen_constants.TCAIREM_OPENAI_API_KEY
        print("OpenAI client initialized successfully!")
        return client
    except Exception as e:
        print(f"An error occurred in initializing OpenAI API: {e}")
        return None
    



def generate_dialogue(note, openai_client):
        note_response = openai_client.chat.completions.create(
                model=gen_constants.dial_generator_config["model"],
                temperature = gen_constants.dial_generator_config["temperature"],
                max_tokens = gen_constants.dial_generator_config["max_tokens"],
                top_p = gen_constants.dial_generator_config["top_p"],
                frequency_penalty = gen_constants.dial_generator_config["frequency_penalty"],
                presence_penalty = gen_constants.dial_generator_config["presence_penalty"],
                messages=[
                    {
                        "role": "system",
                        "content": gen_constants.DIALOGUE_GENERATOR_SYSTEM_PROMPT 
                    },
                    {
                        "role": "user",
                        "content": note
                    }
                ]
                )
        return note_response.choices[0].message.content



NOTE= '''
**1. Subjective:**

- **Chief Complaint:** Severe chronic lower back pain, rated 8/10 in severity, persisting for 3 years.

- **Associated Symptoms:** Muscle spasms, stiffness, fatigue.

- **Frequency:** Daily.

- **Impact on Daily Activities:** Difficulty performing daily tasks, interferes with sleep, limits mobility.

- **Patient's self-reported habits and lifestyle:** 

  - **Sleep:** Reports poor sleep quality and frequent awakenings due to pain.

  - **Diet:** Unbalanced diet, high in processed foods.

  - **Exercise:** Minimal physical activity, avoids exercise due to pain.

  - **Tobacco Use:** Smoker, 1 pack/day.

  - **Alcohol Consumption:** Occasional, 1-2 drinks/week.

  - **Recreational Activities:** Limited due to pain, enjoys reading.

- **Cooperation with Medical Advice:** Moderate, sometimes non-compliant with exercise recommendations.



**2. Objective:**

- **Vital Signs:** Not Documented.

- **Physical Exam:**

  - **Musculoskeletal Examination:** 

    - Decreased range of motion in the lumbar spine.

    - Tenderness on palpation of the paraspinal muscles.

    - Positive straight leg raise test.

- **Investigation/Test Results:** 

  - MRI of the lumbar spine pending, ordered to evaluate possible disc herniation or spinal stenosis.



**3. Assessment:**

- **Diagnosis:** Chronic Pain Syndrome (ICD-10: G89.4)

- **Medical Reasoning:** The patient presents with chronic severe lower back pain persisting for 3 years, accompanied by muscle spasms, stiffness, and fatigue. The physical exam reveals decreased range of motion and tenderness in the lumbar spine. A positive straight leg raise test suggests nerve root involvement. Pending MRI results will help further evaluate for disc herniation or spinal stenosis.



**4. Plan:**

- **Medication Treatment:**

  - Gabapentin 300 mg, oral, once daily.

  - Prednisone 5 mg, oral, once daily for 7 days.

- **Medical Treatment:**

  - Continue current medication regimen with adjustments as prescribed.

- **Referral:**

  - Physical therapy for chronic pain management to Dr. Jessica Black, PT.

- **Lifestyle Recommendations:**

  - Recommend a structured exercise program tailored to pain management.

  - Begin cognitive behavioral therapy (CBT) to help with pain coping strategies.

  - Encourage cessation of smoking and provide resources for smoking cessation support.

  - Suggest dietary modifications to promote a more balanced, nutritious diet.

- **Follow-up:**

  - Follow-up appointment in 4 weeks to assess response to medication adjustments and progress with physical therapy.

- **Patient Education and Counseling:**

  - Discussed the chronic nature of his condition and reinforced the importance of compliance with the treatment plan including medication, physical therapy, and lifestyle modifications.



**Patient Agreements:** The patient understands and agrees with the recommended treatment plan and follow-up schedule.
'''

openai_client= initialize_openai_client()
dial= generate_dialogue(NOTE, openai_client= openai_client)

output_file_path = f'/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/output/dialogues_onVector/ICL_version/test.txt'
with open(output_file_path, 'w') as file:
    file.write(dial)
print(f"Conversation saved to {output_file_path}")
