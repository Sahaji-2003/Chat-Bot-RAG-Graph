# test_gpt2.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Define a simple input
input_text = "Hello, how are you?"

# Tokenize the input
inputs = tokenizer.encode_plus(input_text, return_tensors='pt', truncation=True, max_length=1024, padding='max_length')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Generate a response with temperature and top-p sampling
outputs = model.generate(
    input_ids,
    max_new_tokens=50,
    num_return_sequences=1,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.7,  # Lower temperature for less randomness
    top_p=0.9,        # Use nucleus sampling
    top_k=50          # Consider top 50 tokens
)

# Decode the generated response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input Text: ", input_text)
print("Generated Response: ", response)
