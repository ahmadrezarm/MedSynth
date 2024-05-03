# source: https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client

import transformers
from huggingface_hub import InferenceClient

#TODO: make the token an enviroenment variable
token = "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"

messages = [{"role": "user", "content": "What is the capital of France?"}]
client = InferenceClient("unsloth/llama-3-8b-Instruct-bnb-4bit", token=token)
client.chat_completion(messages, max_tokens=100)

