import json
from openai import OpenAI
from datasets import load_dataset
import random
import os
import hydra
import base64
from formatting import format

# OpenAI API 키 설정
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def check_duplicates(ds):
    def counting(ds):
        prompts = {}
        for i in range(len(ds['train'])):
            prompt = ds['train'][i]["id"]
            if prompt in prompts:
                prompts[prompt] += 1
            else:
                prompts[prompt] = 1
        return prompts
    
    result = counting(ds)
    cnt = 0
    cntsum = 0
    unique = len(result)
    for key, value in result.items():
        if value > 1:
            cnt += 1    
            cntsum += value
    return cnt, cntsum, unique

# generating response tags
def generate_response(cfg, prompt, base64_image):  
    response = client.chat.completions.create(
        model=cfg.gpt_model,
        messages=[
            {"role": "system", "content": "The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, interesting, engaging, educational, and helpful."},
            {"role": "user", "content": [
                {
                    "type": "text", 
                    "text": f"{prompt}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },

                ]}
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content

def make_list(data):
    new_data = []
    for i in range(len(data)):
        new_data.append(data[i])
    return new_data

# Process the response
def process_response(cfg, ds):
    prompts = {} # Create a dictionary to store the prompts
    
    data = ds['train'].to_list()
    print("-----------------------------")
    print("Data Processing Started.")

    random.seed(42)
    if (cfg.sample == True):
        print("Sampling is enabled.")
        sample_size = max(1, (len(data) // 100) * cfg.sample_size)
        print(sample_size)
        sampled_data = random.sample(data, sample_size)
        print("Sample size : " + f"{sample_size}")
    else:
        print("Sampling is disabled.")
        sampled_data = data
    

    # Loop through the data
    prompt = {}
    # 중복인 Prompt가 있을 수 있음.
    for i in range(len(sampled_data)):
        if i > cfg.max_data:
            break
        
        if i < cfg.start_point:
            continue

        prompt = sampled_data[i]["prompt"]
        id = sampled_data[i]['id']
        models = sampled_data[i]['models']
        completions = sampled_data[i]['completions']
        print(f"Prompt: {prompt}")
        
        file_path = os.path.join(cfg.output_dir, f"{id}.jpg")
        image_jpg = encode_image(file_path)

        if prompt in prompts:
            response = prompts[prompt]
        else:
            response = generate_response(cfg, prompt, image_jpg) # Generate the response
            print(response)
            prompts[prompt] = response
        
        new_data = {
            "id": id, 
            "prompt": prompt, 
            "models": models, 
            "completions": completions, 
            "gpt_inference": response
        }
        # format the data
        new_data = format(new_data)

        # save the data
        with open('../results/datasets/gpt_vl_results_30%.jsonl', 'a') as file:
            json.dump(new_data, file, ensure_ascii=False)
            file.write('\n')
            print(f"{i+1}" + "/" + f"{len(sampled_data)}" +" Response appended.")

    
@hydra.main(version_base=None, config_path="./config", config_name="gpt-inference-config")
def main(cfg):
    # Call the function
    # Load the config file
    print(cfg)

    # Load the dataset
    ds = load_dataset(cfg.dataset)
    cnt, cntsum, unique = check_duplicates(ds)
    print("-----------------------------")
    print("Duplicate Data Count :" + f"{cnt}" + "/" +f"{len(ds['train'])}")
    print("Dataset Count Average :" + f"{cntsum/cnt}")
    print("Unique Prompt Count :" + f"{unique}")

    process_response(cfg, ds)
    print("Data Processing Completed.")


if __name__ == "__main__":
    main()