

import json
import random
import os
import hydra 
import pandas as pd

def process_and_split_data(dataset, output_filenames, baseline):
    # average_score로 정렬
    dataset_sorted = sorted(dataset, key=lambda x: x[baseline])

    # 데이터셋 길이 계산
    total_len = len(dataset)
    split1 = total_len // 3  # 첫 번째 33%
    print(f"length of split1: {split1}")
    split2 = 2 * total_len // 3  # 두 번째 33%
    print(f"length of split2: {split2- split1}")
    print(f"length of split3: {total_len - split2}")

    # 33%씩 데이터 나누기
    group1 = dataset_sorted[:split1]
    group2 = dataset_sorted[split1:split2]
    group3 = dataset_sorted[split2:]

    # 그룹의 경계값 출력
    print("33% 경계값:")
    print(f"Group 1 (하위 33%): 최대 {baseline} = {group1[-1][baseline] if group1 else 'N/A'}")
    print(f"Group 2 (중간 33%): 최대 {baseline} = {group2[-1][baseline] if group2 else 'N/A'}")
    print(f"Group 3 (상위 33%): 최대 {baseline} = {group3[-1][baseline] if group3 else 'N/A'}")

    #그룹을 JSON 파일로 저장
    for group, filename in zip([group1, group2, group3], output_filenames):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(group, f, ensure_ascii=False, indent=4)

    print(f"Data successfully split and saved to: {', '.join(output_filenames)}")

def process_and_split_data_no_overlap(dataset, output_filename):
    # DataFrame으로 변환
    data = pd.DataFrame(dataset)
    
    # NaN 제거
    nan_count = data["score_variance"].isna().sum()
    print(f"NaN count in score_variance: {nan_count}")
    
    data = data.dropna(subset=["score_variance"])

    # score_variance 기준으로 정렬
    data_sorted = data.sort_values(by="score_variance")

    # 데이터셋 길이 계산
    total_len = len(data_sorted)
    print(f"Total dataset length: {total_len}")
    
    split1 = total_len // 3  # 첫 번째 33%
    split2 = 2 * total_len // 3  # 두 번째 33%

    print(f"Length of split1: {split1}")
    print(f"Length of split2: {split2 - split1}")
    print(f"Length of split3: {total_len - split2}")

    # 33% variance로 데이터 나누기 (Low variance 그룹)
    high_variance = data_sorted.iloc[split2:]  # High variance
    low_variance = data_sorted.iloc[:split2]  # Low variance

    # 50% average로 데이터 나누기
    data_sorted2 = low_variance.sort_values(by="average_score")

    # 데이터셋 길이 계산
    total_low_len = len(data_sorted2)
    print(f"Total low variance dataset length: {total_low_len}")
        
    split3 = total_low_len // 2
    high_average = data_sorted2.iloc[split3:]  # High average
    low_average = data_sorted2.iloc[:split3]  # Low variance

    # JSONL 파일로 저장
    high_variance.to_json("./full_highvar.jsonl", orient="records", lines=True, force_ascii=False)
    low_variance.to_json("./full_lowvar.jsonl", orient="records", lines=True, force_ascii=False)
    high_average.to_json("./full_highavg.jsonl", orient="records", lines=True, force_ascii=False)
    low_average.to_json("./full_lowavg.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"Data successfully split and saved.")

def process_and_split_data_no_overlap_old(dataset, output_filename):
    # average_score로 정렬
    dataset_sorted = sorted(dataset, key=lambda x: x["score_variance"])
    # 데이터셋 길이 계산
    total_len = len(dataset)
    split1 = total_len // 3  # 첫 번째 33%
    print(f"length of split1: {split1}")
    split2 = 2 * total_len // 3  # 두 번째 33%
    print(f"length of split2: {split2- split1}")
    print(f"length of split3: {total_len - split2}")

    # 33% variance로 데이터 나누기
    high_variance = dataset_sorted[split2:] # High variance

    # 50% average로 데이터 나누기
    dataset_sorted = sorted(dataset_sorted[:split2], key=lambda x: x["average_score"])

    split1 = len(dataset_sorted) // 2
    low_average = dataset_sorted[:split1] # Low average
    high_average = dataset_sorted[split1:] # High average

    output_filenames= ['high_variance.jsonl', 'low_average.jsonl', 'high_average.jsonl']

    for group, filename in zip([high_variance, low_average, high_average], output_filenames):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(group, f, ensure_ascii=False, indent=4)
    print(f"Data successfully split and saved.")

def random_dataset(dataset: list, seed: int) -> list:

    # Set random seed for reproducibility
    random.seed(seed)
    print(f"length of dataset: {len(dataset)}")
    # Calculate the number of samples to select

    # Randomly sample indices vl=8020, ultra=6526
    sampled_indices = random.sample(range(len(dataset)), 6526)

    # Extract the sampled items
    data = [dataset[i] for i in sampled_indices]

    print(f"length of data: {len(data)}")

    # Save the sampled data to a file
    with open(f"ultra_sft3ep_random_sample_{seed}.jsonl", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def corr_dataset(dataset):

    # average_score로 정렬
    dataset_sorted = sorted(dataset, key=lambda x: x["correlation"])

    # 데이터셋 길이 계산
    total_len = len(dataset)
    print(f"length of dataset: {total_len}")

    # 0.6 이하의 correlation이 있는 데이터 추출
    low_corr = [d for d in dataset_sorted if d["correlation"] <= 0.65]
    print(f"length of low_corr: {len(low_corr)}")

    # correlation이 상위 1% 데이터 추출
    top_corr = dataset_sorted[-int(total_len * 0.01):]
    print(f"length of top_corr: {len(top_corr)}")

    # Save the sampled data to a file
    with open(f"top_corr.jsonl", 'w', encoding='utf-8') as f:
        json.dump(top_corr, f, ensure_ascii=False, indent=4)

    low_corr = dataset_sorted[:int(total_len * 0.01)]
    print(f"length of low_corr: {len(low_corr)}")

    with open(f"low_corr.jsonl", 'w', encoding='utf-8') as f:
        json.dump(low_corr, f, ensure_ascii=False, indent=4)



def load_processed_data(data_path: str) -> list:
    with open(data_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data

@hydra.main(version_base=None, config_path="../config", config_name="split-data-config")
def main(cfg):

    # Load the processed data
    
    dataset = load_processed_data(cfg.split_data_path)

    process_and_split_data_no_overlap(dataset, cfg.output_filenames)


if __name__ == "__main__":
    main()