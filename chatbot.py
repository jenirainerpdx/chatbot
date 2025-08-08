from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

MAX_HISTORY = 3  # number of past user-bot exchanges to keep
conversation_history = []

# Determine a safe maximum encoder context length
# Prefer the model's max_position_embeddings if available; otherwise default to 512.
max_ctx = getattr(model.config, "max_position_embeddings", None)
if not isinstance(max_ctx, int) or max_ctx <= 0:
    max_ctx = 512

# Build a function to assemble the model input from history + current user input
def build_input_text(history, user_msg, sep):
    # history is a flat list: [user, bot, user, bot, ...]
    # We join them with the model's EOS token (or newline if EOS is None)
    turns = history + [user_msg] if user_msg else history[:]
    return sep.join(turns).strip()

eos = tokenizer.eos_token or "\n"

while True:
    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user_input:
        continue

    # Assemble input and tokenize with truncation to avoid positional index overflow
    text = build_input_text(conversation_history, user_input, eos)
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_ctx,  # ensure encoder inputs never exceed model's positional limit
    )

    # Generate a response
    outputs = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask"),
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
        errors="replace",
    ).strip()

    print(response)

    # Update history and keep only the last MAX_HISTORY exchanges (user, bot pairs)
    conversation_history.append(user_input)
    conversation_history.append(response)
    conversation_history = conversation_history[-(2 * MAX_HISTORY):]