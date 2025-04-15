import json
from scores.model_prob import proxy_probability
from scores.rouge import get_rouge_score
from scores.embedding import get_embedding

def run_score(cfg, dataset):
    if cfg.score_method == "sft":
        proxy_probability(cfg, dataset)
        return
    elif cfg.score_method == "dpo":
        proxy_probability(cfg, dataset)
        return
    elif cfg.score_method == "st":
        get_embedding(cfg, dataset)
        pass
    elif cfg.score_method == "nv":
        get_embedding(cfg, dataset)
        pass
    elif cfg.score_method == "rouge":
        get_rouge_score(cfg, dataset)
        return
    else:
        assert False, f"Invalid score method: {cfg.score_method}"