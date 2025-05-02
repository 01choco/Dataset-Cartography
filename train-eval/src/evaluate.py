from api.export_mt import write_sheet_data_mt
from api.export_evol import write_sheet_data_evol
from api.export_alpaca import write_sheet_data_alpaca
from api.export_hhh import write_sheet_data_hhh
import os
import subprocess
import threading
import time

def evaluate_model_mt(cfg, i, param):
    beta, lr, ratio, data, ckpt = param
    print(f"MT-Bench Evaluating model using {cfg.model}_{cfg.type}_{data}_{i}.yaml")
    eval_list = cfg.epoch_list.copy()
    if cfg.last_ckpt == True:
        eval_list.append(cfg.last_ckpt_epoch)
    models = [f'{cfg.model}_{cfg.type}_{data}_{i}_epoch{j}' for j in eval_list]
    model_list = ' '.join(models)
    judge_name = f'{cfg.judge_name}'

    cwd = os.getcwd()
    os.chdir("eval/FastChat/fastchat/llm_judge")

    if cfg.parallel_eval is True:
        commands_1 = []
        commands_2 = []
        for idx, model in enumerate(models):
            model_path = f'../../../../LLaMA-Factory/{cfg.model_path}/{model}'
            if idx % 2 == 0:
                inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model}" 
                commands_1.append(inf_command)
            else:
                inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device_2} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model}" 
                commands_2.append(inf_command)

        def run_commands(cmd_list, cuda_visible_device):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)
            for cmd in cmd_list:
                print(f"Running on GPU {cuda_visible_device}: {cmd}")
                subprocess.run(cmd, shell=True)
        
        thread1 = threading.Thread(target=run_commands, args=(commands_1, cfg.judge_device))
        thread2 = threading.Thread(target=run_commands, args=(commands_2, cfg.judge_device_2))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        print("Parallel gen_model_answer done.")
        
        gen_command = f"PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model {judge_name} --model-list {model_list} --parallel 2"
        show_command = f"PYTHONPATH=../.. python show_result.py --judge-model {judge_name} --model-list {model_list}"
        subprocess.run(gen_command, shell=True)
        subprocess.run(show_command, shell=True)

        gen_command = f"PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model {judge_name} --model-list {model_list} --parallel 2 --mode pairwise-baseline"
        show_command = f"PYTHONPATH=../.. python show_result.py --judge-model {judge_name} --model-list {model_list} --mode pairwise-baseline"
        subprocess.run(gen_command, shell=True)
        subprocess.run(show_command, shell=True)
        os.chdir(cwd)
    else:
        for model in models:
            model_path = f'../../../../LLaMA-Factory/{cfg.model_path}/{model}'
            inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model}" 
            subprocess.run(inf_command, shell=True)
        
        gen_command = f"PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model {judge_name} --model-list {model_list} --parallel 2"
        show_command = f"PYTHONPATH=../.. python show_result.py --judge-model {judge_name} --model-list {model_list}"
        subprocess.run(gen_command, shell=True)
        subprocess.run(show_command, shell=True)

        gen_command = f"PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model {judge_name} --model-list {model_list} --parallel 2 --mode pairwise-baseline"
        show_command = f"PYTHONPATH=../.. python show_result.py --judge-model {judge_name} --model-list {model_list} --mode pairwise-baseline"
        subprocess.run(gen_command, shell=True)
        subprocess.run(show_command, shell=True)
        os.chdir(cwd)

    # export result to google sheet
    name = f'{cfg.model}-{cfg.type}'
    write_sheet_data_mt(i, name)

def evaluate_model_evol(cfg, i, param):
    beta, lr, ratio, data, ckpt = param
    print(f"Evol_Instruct Evaluating model using {cfg.model}_{cfg.type}_{data}_{i}.yaml")
    eval_list = cfg.epoch_list.copy()
    if cfg.last_ckpt == True:
        eval_list.append(cfg.last_ckpt_epoch)
    models = [f'{cfg.model}_{cfg.type}_{data}_{i}_epoch{j}' for j in eval_list]
    model_list = ' '.join(models)
    judge_name = f'{cfg.judge_name}'

    cwd = os.getcwd()
    os.chdir("eval/FastChat/fastchat/llm_judge")

    if cfg.parallel_eval is True:
        commands_1 = []
        commands_2 = []
        for idx, model in enumerate(models):
            model_path = f'../../../../LLaMA-Factory/{cfg.model_path}/{model}'
            if idx % 2 == 0:
                inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model} --bench-name evol_instruct" 
                commands_1.append(inf_command)
            else:
                inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device_2} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model} --bench-name evol_instruct" 
                commands_2.append(inf_command)

        def run_commands(cmd_list, cuda_visible_device):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)
            for cmd in cmd_list:
                print(f"Running on GPU {cuda_visible_device}: {cmd}")
                subprocess.run(cmd, shell=True)
        
        thread1 = threading.Thread(target=run_commands, args=(commands_1, cfg.judge_device))
        thread2 = threading.Thread(target=run_commands, args=(commands_2, cfg.judge_device_2))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        print("Parallel gen_model_answer done.")
        
        gen_command = f"PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model {judge_name} --model-list {model_list} --parallel 2 --bench-name evol_instruct"
        show_command = f"PYTHONPATH=../.. python show_result.py --judge-model {judge_name} --model-list {model_list} --bench-name evol_instruct"
        subprocess.run(gen_command, shell=True)
        subprocess.run(show_command, shell=True)

        gen_command = f"PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model {judge_name} --model-list {model_list} --parallel 2 --mode pairwise-baseline --bench-name evol_instruct"
        show_command = f"PYTHONPATH=../.. python show_result.py --judge-model {judge_name} --model-list {model_list} --mode pairwise-baseline --bench-name evol_instruct"
        subprocess.run(gen_command, shell=True)
        subprocess.run(show_command, shell=True)
        os.chdir(cwd)
    else:
        for model in models:
            model_path = f'../../../../LLaMA-Factory/{cfg.model_path}/{model}'
            inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model} --bench-name evol_instruct" 
            subprocess.run(inf_command, shell=True)
        
        gen_command = f"PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model {judge_name} --model-list {model_list} --parallel 2 --bench-name evol_instruct"
        show_command = f"PYTHONPATH=../.. python show_result.py --judge-model {judge_name} --model-list {model_list} --bench-name evol_instruct"
        subprocess.run(gen_command, shell=True)
        subprocess.run(show_command, shell=True)

        gen_command = f"PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model {judge_name} --model-list {model_list} --parallel 2 --mode pairwise-baseline --bench-name evol_instruct"
        show_command = f"PYTHONPATH=../.. python show_result.py --judge-model {judge_name} --model-list {model_list} --mode pairwise-baseline --bench-name evol_instruct"
        subprocess.run(gen_command, shell=True)
        subprocess.run(show_command, shell=True)
        os.chdir(cwd)

    # export result to google sheet
    name = f'{cfg.model}-{cfg.type}'
    write_sheet_data_evol(i, name)

def evaluate_model_alpaca(cfg, i, param):
    beta, lr, ratio, data, ckpt = param
    pass
    # export result to google sheet
    write_sheet_data_alpaca(i, data)

def evaluate_model_hhh(cfg, i, param):
    beta, lr, ratio, data, ckpt = param

    print(f"HHH Evaluating model using {cfg.model}_{cfg.type}_{data}_{i}.yaml")
    eval_list = cfg.epoch_list.copy()
    if cfg.last_ckpt == True:
        eval_list.append(cfg.last_ckpt_epoch)
        models = [f'{cfg.model}_{cfg.type}_{data}_{i}_epoch{j}' for j in eval_list]
        print(models)
    else:
        models = [f'{cfg.model}_{cfg.type}_{data}_{i}_epoch{j+1}' for j in range(3)]
    model_list = ' '.join(models)
    judge_name = f'{cfg.judge_name}'

    cwd = os.getcwd()
    os.chdir("eval/instruct-eval")

    for model in models:

        model_path = f"/data/dataset_cartography/Dataset-Cartography/train-eval/LLaMA-Factory/{cfg.model_path}/{model}"
        inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device} python hhh.py main --model_name causal --model_path {model_path} --load_8bit" 
        subprocess.run(inf_command, shell=True)

    os.chdir(cwd)
    name = f'{cfg.model}-{cfg.type}'
    write_sheet_data_hhh(i, name, models)

def evaluate_model(cfg, i, param):
    if cfg.mt_bench is True:
        evaluate_model_mt(cfg, i, param)

    if cfg.evol_instruct is True:
        evaluate_model_evol(cfg, i, param)

    if cfg.alpaca_eval is True:
        evaluate_model_alpaca(cfg, i, param)

    if cfg.hhh is True:
        evaluate_model_hhh(cfg, i, param)