import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import json

# 1. 확률 분포로 변환
def normalize(data):
    total = sum(data)
    return [x / total for x in data]

def kl_divergence_calculation(cfg, dataset):
    
    for data in dataset:
        
        scores = np.array(data['scores'])
        human_preferences = []

        for j in range(len(data['completions'])):
            if cfg.rating == "fine-grained":
                human_preferences.append((data['completions'][j]['fine-grained_score']))
            elif cfg.rating == "helpfulness":
                human_preferences.append(int(data['completions'][j]['annotations']["helpfulness"]["Rating"]))
        human_preferences = np.array(human_preferences)

        P = normalize(human_preferences)  # 분포 A
        Q = scores            # 분포 B (이미 확률 분포로 가정)

        # 3. 분포 간 차이 평가
        # a. Kullback-Leibler Divergence (KL Divergence)
        # KL(P || Q)
        kl_divergence = entropy(P, Q)
        #e의 지수승에 kl_divergence를 넣어서 계산
        kl_divergence = np.exp(-kl_divergence)
        #print(f"KL Divergence D_KL(P || Q): {kl_divergence:.4f}")

        # b. Jensen-Shannon Divergence (JS Divergence)
        js_divergence = jensenshannon(P, Q, base=2)  # base=2 for bits, base=np.e for nats
        #print(f"Jensen-Shannon Divergence D_JS(P || Q): {js_divergence:.4f}")

        # c. Total Variation Distance (TV Distance)
        tv_distance = 0.5 * np.sum(np.abs(np.array(P) - np.array(Q)))
        #print(f"Total Variation Distance D_TV(P, Q): {tv_distance:.4f}")
        
        data['correlation'] = kl_divergence
        data['average_score'] = np.mean(scores)

    with open(cfg.cal_output_path, 'w', encoding='utf-8') as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')




