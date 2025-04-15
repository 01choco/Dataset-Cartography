import json
from openai import OpenAI
from datasets import load_dataset
import random
import os
import hydra

def load_processed_data(data_path: str) -> list:
    with open(data_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data

def format(data):

    completions = data['completions']
    annotations = completions['annotations']
    response_list = completions['response']

    # prompt key 이름을 instruction으로 변경
    data['instruction'] = data.pop('prompt')
    
    new_completions = []
    for i in range(len(response_list)):
        rationale_list = ["Helpfulness", "Ethical Considerations", "Visual Faithfulness"]
        fine_grained_score = 0
        formatted_annotations = {}

        for rationale in rationale_list:
            rating = int(annotations[i][rationale]["Rating"])
            fine_grained_score += rating
            lower_key = rationale.lower().replace(" ", "_")
            formatted_annotations[lower_key] = {
                "Rating": annotations[i][rationale]["Rating"],
                "Rationale": annotations[i][rationale]["Rationale"]
            }
        fine_grained_score /= len(rationale_list)

        new_completions.append({
            "annotations": formatted_annotations,
            "fine-grained_score": fine_grained_score,
            "response": response_list[i]
        })

        data['completions'] = new_completions
    
    return data

def main():

    input_path = "../results/datasets/gpt_vl_results_10%.jsonl"
    dataset = load_processed_data(input_path)

    for data in dataset:
        completions = data['completions']
        
        annotations = completions['annotations']
        response_list = completions['response']

        # prompt key 이름을 instruction으로 변경
        data['instruction'] = data.pop('prompt')
        
        new_completions = []
        for i in range(len(response_list)):
            rationale_list = ["Helpfulness", "Ethical Considerations", "Visual Faithfulness"]
            fine_grained_score = 0
            formatted_annotations = {}

            for rationale in rationale_list:
                rating = int(annotations[i][rationale]["Rating"])
                fine_grained_score += rating
                lower_key = rationale.lower().replace(" ", "_")
                formatted_annotations[lower_key] = {
                    "Rating": annotations[i][rationale]["Rating"],
                    "Rationale": annotations[i][rationale]["Rationale"]
                }
            fine_grained_score /= len(rationale_list)

            new_completions.append({
                "annotations": formatted_annotations,
                "fine-grained_score": fine_grained_score,
                "response": response_list[i]
            })

            data['completions'] = new_completions

    # # save data
    with open(input_path, 'w') as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


if __name__ == "__main__":
    main()