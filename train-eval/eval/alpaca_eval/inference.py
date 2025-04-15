import json
import torch
import yaml
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaMA-2 model using config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to("cuda").eval()
    return tokenizer, model

def generate(model, tokenizer, instruction: str, input_text: str, config) -> str:
    if input_text:
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = f"{instruction}"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=True,
            temperature=config["temperature"],
            top_p=config["top_p"],
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def main():
    args = parse_args()
    config = load_config(args.config)

    tokenizer, model = load_model(config["model_path"])
    eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    output_path = f'example/{config["generator_name"]}.jsonl'
    with open(output_path, "w", encoding="utf-8") as f:
        pass 
    
    print(f"Generating responses to: example/{config["generator_name"]}.jsonl")

    for example in tqdm(eval_set, desc="Generating responses"):
        instruction = example["instruction"]
        input_text = example.get("input", "")

        output = generate(model, tokenizer, instruction, input_text, config)

        r = {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "generator": config["generator_name"],
            "dataset": "alpaca_eval",
            "datasplit": "eval"
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
