import yaml
from api.api import get_sheet_data
from evaluate import evaluate_model
import copy
import subprocess
import hydra
import os
import sys

def create_yaml(cfg, i, param, template):
    beta, lr, ratio, data = param
    if cfg.type == "dpo":
        print(f"Creating {cfg.model}_{cfg.type}_{data}_{i}.yaml with beta={beta}, lr={lr}, data={data}")
    
    elif cfg.type == "simpo":
        print(f"Creating {cfg.model}_{cfg.type}_{data}_{i}.yaml with beta={beta}, lr={lr}, ratio={ratio}, data={data}")

    new = copy.deepcopy(template)
    new['pref_beta'] = float(beta)
    new['learning_rate'] = float(lr)
    new['dataset'] = data
    new['output_dir'] = f'{cfg.save_path}/{cfg.model}-{cfg.type}-{data}-{i}'
    if cfg.type == "simpo":
        new['simpo_gamma'] = float(ratio)
    
    filename = f'./LLaMA-Factory/{cfg.yaml_path}/{cfg.model}_{cfg.type}_{data}_{i}.yaml'

    # save .yaml files
    with open(filename, 'w') as f:
        yaml.dump(new, f, sort_keys=False)

    print(f"Created .yaml file: {filename}")

def train_model(cfg, i, param):
    beta, lr, ratio, data = param
    print(f"Training model using {cfg.model}_{cfg.type}_{data}_{i}.yaml")

    cwd = os.getcwd()
    os.chdir("LLaMA-Factory")

    command = f"CUDA_VISIBLE_DEVICES={cfg.avail_devices} PYTHONPATH=./src llamafactory-cli train {cfg.yaml_path}/{cfg.model}_{cfg.type}_{data}_{i}.yaml"
    subprocess.run(command, shell=True)

    os.chdir(cwd)

def export_model(cfg, i, param, template):
    beta, lr, ratio, data = param
    print(f"Exporting model using {cfg.model}_{cfg.type}_{data}_{i}.yaml")

    cwd = os.getcwd()
    os.chdir("LLaMA-Factory")

    checkpoint_list = cfg.checkpoint_list
    adapter_name = f'{cfg.model}-{cfg.type}-{data}-{i}'
    adapter_path = f'{cfg.save_path}/{adapter_name}'

    for j, checkpoint in enumerate(checkpoint_list):
        print(f"Creating {cfg.model}_{cfg.type}_{data}_{i}_epoch{j+1}.yaml")
        new = copy.deepcopy(template)
        new['adapter_name_or_path'] = f'{adapter_path}/checkpoint-{checkpoint}'
        new['export_dir'] = f'{cfg.model_path}/{cfg.model}_{cfg.type}_{data}_{i}_epoch{j+1}'

        filename = f'{cfg.export_yaml_path}/{cfg.model}_{cfg.type}_{data}_{i}_epoch{j+1}.yaml'

        # YAML 저장
        with open(filename, 'w') as f:
            yaml.dump(new, f, sort_keys=False)

    full_j = len(checkpoint_list)
    print(f"Creating {cfg.model}_{cfg.type}_{data}_{i}_epoch{full_j+1}.yaml")
    new = copy.deepcopy(template)
    new['adapter_name_or_path'] = f'{adapter_path}'
    new['export_dir'] = f'{cfg.model_path}/{cfg.model}_{cfg.type}_{data}_{i}_epoch{full_j+1}'

    filename = f'{cfg.export_yaml_path}/{cfg.model}_{cfg.type}_{data}_{i}_epoch{full_j+1}.yaml'

    # yaml file save 
    with open(filename, 'w') as f:
        yaml.dump(new, f, sort_keys=False)
    
    for k in range(len(checkpoint_list) + 1):
        export_command = f"CUDA_VISIBLE_DEVICES={cfg.avail_devices} PYTHONPATH=./src llamafactory-cli export {cfg.export_yaml_path}/{cfg.model}_{cfg.type}_{data}_{i}_epoch{k+1}.yaml"
        subprocess.run(export_command, shell=True)
        pass

    os.chdir(cwd)

def delete_model(cfg, i, param):
    beta, lr, ratio, data = param
    print(f"Deleting model using {cfg.model}_{cfg.type}_{data}_{i}.yaml")
    models = [f'{cfg.model}_{cfg.type}_{data}_{i}_epoch{j+1}' for j in range(3)]
    for model in models:
        del_path = f'./LLaMA-Factory/{cfg.model_path}/{model}'
        command = f"rm -r {del_path}"
        subprocess.run(command, shell=True)
        print(f"Deleted {del_path}.")

@hydra.main(version_base=None, config_path=".", config_name="hyper-config")
def main(cfg):
    values = get_sheet_data({cfg.model}, {cfg.type})
    print(values)
    with open(f'./LLaMA-Factory/{cfg.yaml_path}/template.yaml', 'r') as f:
        template = yaml.safe_load(f)
        
    with open(f'./LLaMA-Factory/{cfg.export_yaml_path}/template.yaml', 'r') as f:
        export_template = yaml.safe_load(f)
        
    for i,param in enumerate(values):
        if cfg.start <= i and cfg.end >= i:
            if cfg.train == True:
                create_yaml(cfg, i, param, template)
                train_model(cfg, i, param)
            if cfg.export == True:
                export_model(cfg, i, param, export_template)
            if cfg.eval == True:
                evaluate_model(cfg, i, param)
            if cfg.export == True:
                delete_model(cfg, i, param)

if __name__ == "__main__":
    main()