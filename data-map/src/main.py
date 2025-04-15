
import json
import random
import os
import hydra 
from scores.score import run_score
from calculate.variance import run_variance
from calculate.correlation import run_correlation
from figure.figure import get_figure

def load_processed_data(data_path: str) -> list:
    with open(data_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data

@hydra.main(version_base=None, config_path="./config", config_name="main-config")
def main(cfg):
    objective = cfg.objective
    
    if objective == "score":
        dataset = load_processed_data(cfg.score_input_path)
        run_score(cfg, dataset)

    elif objective == "calculation":
        dataset = load_processed_data(cfg.cal_input_path)
        variance_data = run_variance(cfg, dataset)
        run_correlation(cfg, variance_data)

    elif objective == "figure":
        dataset = load_processed_data(cfg.fig_input_path)
        get_figure(cfg, dataset)
        pass
    else:
        assert False, f"Invalid objective: {objective}"

    return

if __name__ == "__main__":
    main()