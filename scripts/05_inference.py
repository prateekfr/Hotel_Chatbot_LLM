import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM

# ========== Config ==========
BASE_MODEL_PATH = "tiiuae/Falcon3-1B-Instruct"
LORA_ADAPTER_PATH = "model/falcon-lora-finetuned"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading Falcon base model...")
    os.makedirs("offload", exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        offload_folder="offload"  # Use `offload_folder`, not `offload_dir`
    )

    print("🔗 Applying LoRA adapter with offload...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH,
        device_map="auto",
        offload_folder="offload"
    )

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def main():
    model, tokenizer = load_model_and_tokenizer()

    print("\n Model is ready! Ask your hotel-related question.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("🧾 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        response = generate_response(model, tokenizer, prompt)
        print(f"🤖 Bot: {response}\n")

if __name__ == "__main__":
    main()
