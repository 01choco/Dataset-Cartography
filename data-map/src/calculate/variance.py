import hydra
import json
import numpy as np
from itertools import combinations

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def proxy_variance_calculation(cfg, dataset):
    
    output_data = []
    for data in dataset:   
        score_variance = np.var(data['scores'])
        data['score_variance'] = score_variance
        output_data.append(data)

    return output_data

def response_variance_calculation(cfg, dataset):
    if cfg.score_method == "st":
        model = SentenceTransformer('all-mpnet-base-v2')
    elif cfg.score_method == "nv":
        model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
    
    for item in dataset:
        
        responses = [item["completions"][j]["response"] for j in range(len(item["completions"]))]
        response_pairs = list(combinations(responses, 2))

        similarity_scores = []
        for pair in response_pairs:
            embeddings = model.encode(pair)
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
            similarity_scores.append(similarity)

        score_variance = np.var(similarity_scores)
        item['score_variance'] = float(score_variance)

    return dataset


def run_variance(cfg, dataset):
    if cfg.var_method == "proxy":
        var_dataset = proxy_variance_calculation(cfg, dataset)
        return var_dataset
    
    elif cfg.var_method == "comb" and (cfg.score_method == "st" or cfg.score_method == "nv"):
        var_dataset = response_variance_calculation(cfg, dataset)
        return var_dataset
    else:
        print("ERROR : Invalid Variance calculation method")
