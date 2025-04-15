from itertools import combinations
import numpy as np
import json

def kendalls_tau(x, y):
    if len(x) != len(y):
        raise ValueError("ERROR : Vectors must have the same length.")

    n = len(x)
    concordant = 0 #C
    discordant = 0 #D

    # Iterate over all pairs of elements
    for (i, j) in combinations(range(n), 2):
        # Compare the relative ranks of the two pairs
        pair1 = (x[i] - x[j]) * (y[i] - y[j])
        if pair1 > 0:
            concordant += 1
        elif pair1 < 0:
            discordant += 1

    # Calculate Kendall's Tau
    tau = (concordant - discordant) / (0.5 * n * (n - 1))
    return tau

def add_jitter(vector, epsilon=1e-9):
    noise = np.random.uniform(-epsilon, epsilon, len(vector))
    return (np.array(vector) + noise).tolist()

def convert_to_rank(vector):
    sorted_indices = np.argsort(vector)
    ranks = np.empty_like(sorted_indices, dtype=float)
    sorted_vector = np.array(vector)[sorted_indices]

    i = 0
    while i < len(sorted_vector):
        # Find indices of ties
        tie_start = i
        while i + 1 < len(sorted_vector) and sorted_vector[i] == sorted_vector[i + 1]:
            i += 1
        tie_end = i

        # Assign average rank for ties
        average_rank = (tie_start + tie_end + 2) / 2.0
        for j in range(tie_start, tie_end + 1):
            ranks[sorted_indices[j]] = average_rank

        i += 1

    return ranks.tolist()

def kendalls_tau_calculation(cfg, dataset):

    for item in dataset:

        # set similarity scores
        scores = item["scores"]
        score_rank = convert_to_rank(scores)

        annotations = []
        for j in range(len(item['completions'])):
            if cfg.rating == "fine-grained":
                annotations.append((item['completions'][j]['fine-grained_score']) / 5)
            elif cfg.rating == "helpfulness":
                annotations.append(int(item['completions'][j]['annotations']["helpfulness"]["Rating"]) / 5)
        
        ann = add_jitter(annotations)
        ann_rank = convert_to_rank(ann)
        tau = kendalls_tau(score_rank, ann_rank)

        # save the similarity scores
        item['correlation'] = tau
        item['average_score'] = np.mean(scores)

    with open(cfg.cal_output_path, 'w', encoding='utf-8') as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
