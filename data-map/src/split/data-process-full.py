import json
import hydra
from datasets import load_dataset

def convert_data(cfg, train):
    cnt = 0
    for i in range(len(train)):
        item = train[i]
        
        # set response data
        data_input = item["instruction"]
        response = {}

        for j in range(len(item["completions"])):
            generated_response = item["completions"][j]["response"]
            if (cfg.type == "help"):
                response[generated_response] = item['completions'][j]['annotations']["helpfulness"]["Rating"]
            elif (cfg.type == "fine"):
                response[generated_response] = item['completions'][j]["fine-grained_score"]
        
        new_datas = []

        keys = list(response.keys())
        for j in range(len(keys)):
            for k in range(j + 1, len(keys)):
                key = keys[j]
                value = response[key]
                key2 = keys[k]
                value2 = response[key2]

                if key == key2:
                    continue
                else:
                    if value > value2:
                        chosen = key
                        rejected = key2
                    elif value < value2:
                        chosen = key2
                        rejected = key
                    elif value == value2:
                        continue
                    new_data = {
                        "instruction": data_input,
                        "input": "",
                        "chosen": chosen,
                        "rejected": rejected
                    }
                    new_datas.append(new_data)
        
        for new_data in new_datas:
            with open(cfg.output_path, 'a') as file:
                json.dump(new_data, file)
                file.write('\n')
            #print(f"{len(new_datas)} data appended. {i+1} / {len(train)} Data converted.")
        cnt += len(new_datas)
    return cnt 

@hydra.main(version_base=None, config_path=".", config_name="ppo-full-config")
def main(cfg):
    print(cfg)

    data = load_dataset("json", data_files=cfg.input_path)
    print("-----------------------------")
    print("Data Conversion started.")

    cnt = convert_data(cfg, data["train"])
    print(f"Data Processing completed. {cnt} data converted and appended.")

if __name__ == "__main__":
    main()
