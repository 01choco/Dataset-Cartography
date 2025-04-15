import json
from openai import OpenAI
from datasets import load_dataset
import random
import os
import hydra 

# OpenAI API 키 설정
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def check_duplicates(ds):
    def counting(ds):
        prompts = {}
        for i in range(len(ds['train'])):
            prompt = ds['train'][i]["instruction"]
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
def generate_response(cfg, prompt):  
    response = client.chat.completions.create(
        model=cfg.gpt_model,
        messages=[
            {"role": "system", "content": "The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, interesting, engaging, educational, and helpful."},
            {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content

# Check helpfulness data
def check_helpfulness(data):
    new_data = []
    for i in range(len(data)):
        helpfulcount = 0
        for j in range(len(data[i]["completions"])):
            principle = data[i]["completions"][j]["principle"]
            if "helpfulness" == principle:
                helpfulcount += 1
        if helpfulcount == 4:
            new_data.append(data[i])
    print("-----------------------------")
    print("Helpfulness Data Deteceted : " + f"{len(new_data)}")
    return new_data
            

# Process the response
def process_response(cfg, ds):
    prompts = {} # Create a dictionary to store the prompts
    
    # Check the helpfulness of the data
    helpfulness_data = check_helpfulness(ds['train'])
    print("-----------------------------")
    print("Data Processing Started.")

    # Select 1% of the helpfulness data randomly
    random.seed(42)
    if (cfg.sample == True):
        print("Sampling is enabled.")
        sample_size = max(1, (len(helpfulness_data) // 100) * cfg.sample_size)
        sampled_data = random.sample(helpfulness_data, sample_size)
        print("Sample size : " + f"{sample_size}")
    else:
        print("Sampling is disabled.")
        sampled_data = helpfulness_data
    
    # Loop through the data
    for i in range(len(sampled_data)):
        if i > cfg.max_data:
            break
        
        if i < cfg.start_point:
            continue

        prompt = sampled_data[i]["instruction"]
        if prompt in prompts:
            response = prompts[prompt]
        else:
            response = generate_response(cfg, prompt) # Generate the response
            prompts[prompt] = response
        
        
        new_data = sampled_data[i]
        new_data['gpt_inference'] = response

            # Write the processed data to a new JSONL file
        with open('./results/result.jsonl', 'a') as file:
            json.dump(new_data, file)
            file.write('\n')
            print(f"{i+1}" + "/" + f"{len(sampled_data)}" +" Response appended.")

@hydra.main(version_base=None, config_path="./config", config_name="gpt-inference-config")
def main(cfg):
    # Call the function
    # Load the config file
    print(cfg)

    # Load the dataset
    ds = load_dataset(cfg.dataset)

    # Check for duplicates
    cnt, cntsum, unique = check_duplicates(ds)
    print("-----------------------------")
    print("Duplicate Data Count :" + f"{cnt}" + "/" +f"{len(ds['train'])}")
    print("Dataset Count Average :" + f"{cntsum/cnt}")
    print("Unique Prompt Count :" + f"{unique}")

    process_response(cfg, ds)
    print("Data Processing Completed.")


if __name__ == "__main__":
    main()