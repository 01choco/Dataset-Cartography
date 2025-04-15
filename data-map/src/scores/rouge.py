import torch
import json
import hydra
from typing import List, Tuple
from rouge_score import rouge_scorer


def comput_rouge(generated_text, reference_text):

    rouge_scores = []

    # ROUGE scorer 생성 (ROUGE-1, ROUGE-2, ROUGE-L)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # ROUGE 점수 계산
    scores = scorer.score(reference_text, generated_text)

    rouge_scores.append(scores['rougeL'])

    return (scores['rougeL'].fmeasure)

def get_rouge_score(cfg, dataset):

    count = 1
    output_data = []
    
    for data in dataset:
        print(count)
        count += 1

        golden_response = data["gpt_inference"]
        scores = []

        for model_idx, model_alias in enumerate(data["models"]):
            response = data["completions"][model_idx]
            generated_response = response["response"]
            rouge_scores = comput_rouge(generated_response,golden_response)
            scores.append(rouge_scores)
        
        data['scores'] = scores
        output_data.append(data)

    with open(f"{cfg.score_output_path}", 'w', encoding='utf-8') as f:
        for data in output_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')






