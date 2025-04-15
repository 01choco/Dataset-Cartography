import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import List, Tuple
import os
import numpy as np
import torch.nn.functional as F
MACRO_INF = -1e100

def load_processed_data(data_path: str) -> list:
    with open(data_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data

def compute_response_probability(model, tokenizer, prompt: str, response: str) -> float:
    """
    P(response | prompt)를 계산하는 함수.
    return 값은 (확률, 로그확률) 튜플로 구성.
    """
    
    with torch.no_grad():
        if len(response) == 0:
            return 0.0, MACRO_INF
        
        # (a) 토큰화: prompt, response
        prompt_ids   = tokenizer(prompt,   return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

        # (b) 앞에 prompt 토큰 + response 토큰을 이어붙여 input_ids 구성
        #     (주의) response의 첫 번째 토큰과 prompt의 마지막 토큰이 중복된 BOS/패딩 토큰이 없는지 확인
        #     일반적으로는 그냥 concat해도 문제 없지만, 필요하면 아래 예시처럼 처리:
        #       response_ids[:, 0]이 <BOS>일 경우 제거하기 위해 response_ids[:, 1:] 등
        #     여기서는 add_special_tokens=False로 했으므로 직접 확인.
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)

        # (c) labels 설정
        #     -> prompt 부분은 학습(로스) 계산에서 제외하고 싶으므로 -100으로 마스킹
        labels = input_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100

        # (d) 모델 전파 (forward)
        outputs = model(input_ids, labels=labels)
        # outputs.loss는 (전체 response 토큰들)에 대한 평균 cross-entropy
        # ex) loss = - (1/T) * Σ logP(응답토큰|앞부분)
        ce_loss = outputs.loss  # shape: scalar

        # (e) 길이 미고려 per-token log prob
        #     avg_log_prob = (1/T) * sum(log p_i) = - ce_loss
        avg_log_prob = -ce_loss
        per_token_prob = torch.exp(avg_log_prob).item()

    return per_token_prob, avg_log_prob.item()


def compute_response_probability_batch(
    model, 
    tokenizer, 
    prompts: list[str], 
    responses: list[str]
):
    """
    여러 (prompt, response) 쌍을 한 번에 처리하여
    각 샘플의 (확률, 로그확률)을 리스트로 반환.
    
    - prompts[i] 와 responses[i]가 1:1 대응한다고 가정
    - Causal LM(GPT 계열)에서, 각 token_t의 logit은 logits[:, t-1]에서 얻음
    """

    device = next(model.parameters()).device

    # 1) prompt, response 각각 배치 토크나이즈 (add_special_tokens=False 주의)
    #    서로 길이가 다르므로, 나중에 수작업으로 cat할 예정
    prompt_tok = tokenizer(
        prompts, 
        add_special_tokens=True, 
        padding=True,  # 일단 개별로
        truncation=True,
        return_tensors='pt'
    )
    response_tok = tokenizer(
        responses, 
        add_special_tokens=True, 
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # 2) 각 샘플별로 prompt_ids + response_ids를 이어붙임
    #    (길이가 제각각이므로, 파이썬 리스트로 모은 뒤 최종 패딩)
    input_ids_list = []
    prompt_lengths = []
    total_lengths = []

    batch_size = len(prompts)
    for i in range(batch_size):
        # i번째 sample
        p_ids = prompt_tok.input_ids[i].tolist()
        r_ids = response_tok.input_ids[i].tolist()
        cat_ids = p_ids + r_ids  # 단순 연결
        input_ids_list.append(cat_ids)
        
        prompt_lengths.append(len(p_ids))
        total_lengths.append(len(cat_ids))

    # 3) 최대 길이를 구해 pad_token_id로 패딩
    #    (주의) GPT 계열에서 pad_token_id가 없으면 bos_token_id 등으로 처리해야 할 수도 있음
    max_len = max(len(seq) for seq in input_ids_list)
    if tokenizer.pad_token_id is None:
        # GPT2 등은 기본 pad_token_id가 없으니 강제로 설정 (예: 50256)
        tokenizer.pad_token_id = tokenizer.eos_token_id

    padded_input_ids = []
    for seq in input_ids_list:
        pad_len = max_len - len(seq)
        seq_padded = seq + [tokenizer.pad_token_id] * pad_len
        padded_input_ids.append(seq_padded)

    input_ids_pt = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
    attention_mask = (input_ids_pt != tokenizer.pad_token_id).long()

    # 4) 모델 forward (logits만 받음)
    with torch.no_grad():
        outputs = model(input_ids_pt, attention_mask=attention_mask)
        # shape: (batch_size, max_len, vocab_size)
        logits = outputs.logits

    # 5) 각 샘플별로 response 구간의 평균 로그확률 계산
    probs, log_probs = [], []

    for i in range(batch_size):
        p_len = prompt_lengths[i]
        t_len = total_lengths[i]
        r_len = t_len - p_len  # response 길이

        # 응답이 비었다면
        if r_len <= 0:
            probs.append(0.0)
            log_probs.append(float('-inf'))
            continue

        # token t 의 예측 확률은 logits[i, t-1]로부터 얻음 (causal LM)
        # sum(log p(token_t)) over t in [p_len .. t_len-1]
        sum_log_p = 0.0
        for t in range(p_len, t_len):
            if t == 0:
                # t=0인 경우는 이전 토큰이 없다.
                # 만약 prompt 자체가 길이가 0이라면, 첫 토큰은 <BOS> 예측 등 처리.
                # 여기서는 간단히 스킵하거나, 별도 로직을 추가할 수 있음
                continue
            
            # 정답 토큰 ID
            token_id = input_ids_pt[i, t].item()
            # 해당 위치의 logits은 logits[i, t-1, :]
            log_probs_t = F.log_softmax(logits[i, t-1, :], dim=-1)
            sum_log_p += log_probs_t[token_id].item()

        avg_log_p = sum_log_p / r_len
        prob = np.exp(avg_log_p)

        probs.append(prob)
        log_probs.append(avg_log_p)

    return probs, log_probs


def proxy_probability_batch(cfg, dataset, batch_size=4):
    save_file = cfg.score_output_path

    # 1) 모델 & 토크나이저 로드
    model_name = cfg.model_name
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    
    # 데이터 시작 인덱스 처리
    start_idx = cfg.start_idx 
    count = 1

    # (A) dataset에서 필요한 부분만 슬라이싱(혹은 그냥 range로 루프)
    data_slice = dataset[start_idx-1 :]

    for data in data_slice:

        count += 1

        golden_response = data["gpt_inference"]
        prompt = data["instruction"]
        scores = []

        prompts_batch = []
        responses_batch = []
        lengths = []  # 각 item이 가진 model_response 개수 (scores 나눠주기 위해)
        
        model_responses = [resp["response"] for resp in data["completions"]]
        lengths.append(len(model_responses))

        # prompt_j를 model_responses 개수만큼 반복
        for mr in model_responses:
            prompts_batch.append(prompt)
            responses_batch.append(mr)

        # 2) 한 번에 배치로 확률 계산
        model_probs, model_logprobs = compute_response_probability_batch(
            model, tokenizer, prompts_batch, responses_batch
        )

        data["scores"] = model_probs
        # 3) 계산된 결과를 다시 batch 각 아이템별로 분배

        # ----- (C) 이번 배치 결과를 바로 파일에 append 모드로 저장 -----
        with open(f"{save_file}", 'a', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')
            print(f"Response #{count} probabilities appended.")



def proxy_probability(cfg, dataset):
    #0) 만약 저장하려는 파일이 이미 존재한다면 삭제
    save_file = cfg.score_output_path
    if os.path.exists(save_file):
        os.remove(save_file)

    # 1) 모델 & 토크나이저 로드
    model_name = cfg.model_name
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    count = 1
    
    for data in dataset:

        if count < cfg.start_idx - 1:
            count += 1
            continue
        elif count > cfg.end_idx:
            break
        count += 1

        golden_response = data["gpt_inference"]
        prompt = data["instruction"]
        scores = []

        # 3) gold response probability 계산
        gold_prob, gold_logprob = compute_response_probability(model, tokenizer, prompt, golden_response)
        data['gold prob'] = gold_prob

        #4) model response probability 계산
        for model_idx, model_alias in enumerate(data["models"]):
            completions = data["completions"]
            generated_response = completions[model_idx]["response"]
            prob, logprob = compute_response_probability(model, tokenizer, prompt, generated_response)
            scores.append(prob)
        
        data['scores'] = scores

        with open(save_file, 'a', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')
            print(f"Response #{count} probabilities appended.")
