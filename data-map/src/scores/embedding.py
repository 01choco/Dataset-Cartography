import json
import hydra
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def get_embedding(cfg, dataset):
    if cfg.score_method == "st":
        model = SentenceTransformer('all-mpnet-base-v2')
    elif cfg.score_method == "nv":
        model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)

    output_data = []
    for item in dataset:

        # set response data
        golden_response = item["gpt_inference"]
        similarity_scores = []

        for j in range(len(item["completions"])):
            generated_response = item["completions"][j]["response"]
            embeddings = model.encode([golden_response, generated_response])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
            similarity_scores.append(similarity)
        
        similarity_scores_list = [float(arr[0][0]) for arr in similarity_scores]


        item['scores'] = similarity_scores_list
        output_data.append(item)

    with open(f"{cfg.score_output_path}", 'w', encoding='utf-8') as f:
        for data in output_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


