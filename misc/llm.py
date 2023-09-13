from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load a smaller T5 model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Input text
input_text = "Translate the following English text to French: 'Hello, how are you?'"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate the output
output = model.generate(input_ids)

# Decode and print the output text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Output Text:", output_text)
