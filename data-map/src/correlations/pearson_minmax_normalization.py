import numpy as np
import json

def pearson_minmax_calculation(cfg, dataset):

    for item in dataset:
        
        # set similarity scores
        scores = np.array(item["scores"])

        annotations = []

        for j in range(len(item['completions'])):
            if cfg.rating == "fine-grained":
                annotations.append((item['completions'][j]['fine-grained_score']) / 5)
            elif cfg.rating == "helpfulness":
                annotations.append(int(item['completions'][j]['annotations']["helpfulness"]["Rating"]) / 5)

        human_scores = np.array(annotations)
        
        # Min-Max 정규화
        cosine_normalized = (scores - scores.min()) / (scores.max() - scores.min())
        human_normalized = (human_scores - human_scores.min()) / (human_scores.max() - human_scores.min())

        # 피어슨 상관계수 계산
        pearson_corr = np.corrcoef(cosine_normalized, human_normalized)[0, 1]
        
        item['correlation'] = pearson_corr
        item['average_score'] = np.mean(scores)

    with open(cfg.cal_output_path, 'w', encoding='utf-8') as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')