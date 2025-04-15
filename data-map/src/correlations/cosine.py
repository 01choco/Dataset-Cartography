import numpy as np
import json

def cosine_calculation(cfg, dataset):

    output_data = []
    for data in dataset:
        # set probabilities
        scores = np.array(data['scores'])

        # set annotations
        annotations = []
        for j in range(len(data['completions'])):
            if cfg.rating == "fine-grained":
                    annotations.append((data['completions'][j]['fine-grained_score']))
            elif cfg.rating == "helpfulness":
                    annotations.append(int(data['completions'][j]['annotations']["helpfulness"]["Rating"]))
        ann_scores = np.array(annotations)

        # 코사인 유사도 계산
        cosine_similarity = np.dot(scores, ann_scores) / (np.linalg.norm(scores) * np.linalg.norm(ann_scores))

        # 결과 출력
        new_data = data
        new_data['correlation'] = cosine_similarity
        new_data['average_score'] = np.mean(scores)

        output_data.append(new_data)

    with open(cfg.cal_output_path, 'w', encoding='utf-8') as f:
        for data in output_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
