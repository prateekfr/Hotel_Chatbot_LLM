# Save as scripts/convert_to_instruct.py
import json

input_file = "data/processed/processed_dataset.json"
output_file = "data/processed/falcon_finetune_dataset.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

with open(output_file, "w", encoding="utf-8") as out:
    for item in qa_data:
        prompt = f"### Instruction:\n{item['question']}\n\n### Response:\n{item['answer']}"
        out.write(json.dumps({"text": prompt}) + "\n")

print(f"Converted to fine-tuning format: {output_file}")
