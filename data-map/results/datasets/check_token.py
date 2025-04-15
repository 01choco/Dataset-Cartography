import json
import random
import os
from transformers import AutoTokenizer

def load_processed_data(data_path: str) -> list:
    with open(data_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data



def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # 원하는 모델로 변경 가능
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 사용 예
    data_path = './vlfeedback_80k.jsonl'
    data = load_processed_data(data_path)

    print(f"Loaded {len(data)} records.")

    for da in data:
        input_tokens = tokenizer.tokenize(da['prompt'])
        gpt_4v_response = da['gpt_inference']
        output_tokens = tokenizer.tokenize(gpt_4v_response)
        print(f"Input tokens: {len(input_tokens)}")
        print(f"Output tokens: {len(output_tokens)}")


if __name__ == "__main__":
    main()