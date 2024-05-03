import torch
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)

if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)




Aci_test = pd.read_csv("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/input/clinicalnlp_taskC_test2_SOAP.csv")

i = 0
for conversation in Aci_test["dialogue"]:
    if i == 0:
        messages = [
            {"role": "system", "content": """You are a an assistant to medical doctors and help them summarize their conversations with patients.
            The doctor will give you the conversation and you should summarize it into SOAP format. Please make sure it is comprehensive and accuarate. The summary will be used to
            as the medical note of the visit in Electronic Health Record system."""},
            {"role": "user", "content": f"""{conversation}"""},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=2000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))
