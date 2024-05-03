summarizer_system_prompt = """You are a an assistant to medical doctors and help them summarize their conversations with patients.
                The doctor will give you the conversation and you should summarize it into SOAP format. Please make sure it is comprehensive and accuarate. The summary will be used to
                as the medical note of the visit in Electronic Health Record system."""

base_model= "meta-llama/Meta-Llama-3-8B-Instruct"

Aci_test_path = "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/clinicalnlp_taskC_test2_SOAP.csv"
Aci_train_path = "/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/TaskC-TrainingSet_SOAP.csv"