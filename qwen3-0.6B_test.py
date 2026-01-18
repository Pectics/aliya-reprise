import sys
import torch
import msvcrt
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)
model.eval()

END_THINK_TOKEN_ID = 151668


def read_key():
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        ch2 = msvcrt.getwch()
        if ch2 == "M":
            return "RIGHT"
        return "SPECIAL"
    if ch == "\r":
        return "ENTER"
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch == "\x1b":
        return "ESC"
    return ch


def strip_think_tags(text):
    if not text:
        return text
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    return text


def step_generate(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask

    past_key_values = None
    next_input_ids = None
    first_step = True
    in_think = True
    thinking_text = ""
    content_text = ""

    printed_think_header = False
    printed_content_header = False
    intervention_buffer = []
    in_intervention = False

    def print_stream(piece):
        nonlocal printed_think_header, printed_content_header
        if in_think:
            if not printed_think_header:
                print("thinking:", end=" ", flush=True)
                printed_think_header = True
            print(piece, end="", flush=True)
        else:
            if not printed_content_header:
                if printed_think_header:
                    print("", flush=True)
                print("content:", end=" ", flush=True)
                printed_content_header = True
            print(piece, end="", flush=True)

    def apply_intervention(text_input):
        nonlocal past_key_values, next_input_ids, thinking_text, content_text
        if not text_input:
            return
        if not in_think:
            print("\nIntervention ignored: thinking finished.")
            return
        token_ids = tokenizer.encode(text_input, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=token_ids, use_cache=True, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        next_input_ids = token_ids[:, -1:]

        display_text = strip_think_tags(text_input)
        thinking_text += display_text
        print_stream(display_text)

    print("Step mode: Right Arrow = next token; type to intervene; Enter = apply; Esc = cancel input; Ctrl+C = stop.")

    while True:
        key = read_key()

        if key == "RIGHT":
            if in_intervention and intervention_buffer:
                # Ignore step until buffer applied or canceled
                continue

            with torch.no_grad():
                if first_step:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                    first_step = False
                else:
                    outputs = model(
                        input_ids=next_input_ids,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    past_key_values = outputs.past_key_values
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

            token_id = next_token.item()
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            if token_id == END_THINK_TOKEN_ID:
                in_think = False
                continue

            next_input_ids = next_token.unsqueeze(0)
            piece = tokenizer.decode(next_token, skip_special_tokens=True)
            piece = strip_think_tags(piece)

            if in_think:
                thinking_text += piece
            else:
                content_text += piece
            print_stream(piece)

        elif key == "ENTER":
            if intervention_buffer:
                text_input = "".join(intervention_buffer)
                print("")
                apply_intervention(text_input)
                intervention_buffer = []
                in_intervention = False

        elif key == "ESC":
            if in_intervention:
                intervention_buffer = []
                in_intervention = False
                print("\nIntervention canceled.")

        elif key == "SPECIAL":
            continue

        else:
            if not in_intervention:
                print("\nintervene> ", end="", flush=True)
                in_intervention = True
            if key == "\b":
                if intervention_buffer:
                    intervention_buffer.pop()
                    print("\b \b", end="", flush=True)
            else:
                intervention_buffer.append(key)
                print(key, end="", flush=True)

    print("", flush=True)
    return thinking_text, content_text


def main():
    messages = []
    print("Chat ready. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        try:
            thinking_content, content = step_generate(messages)
        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            continue

        if thinking_content:
            print("\n[thinking preview stored]")
        if content:
            messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    main()
