from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

conversation_history = []

while True:
    history_string = "\n".join(conversation_history)
    user_input = input("> ")
    inputs = tokenizer.encode_plus(
        history_string,
        user_input,
        return_tensors="pt",
    )
    outputs = model.generate(**inputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(response)
    conversation_history.append(user_input)
    conversation_history.append(response)