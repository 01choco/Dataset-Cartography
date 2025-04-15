from api.export_mt import write_sheet_data_mt
from api.export_evol import write_sheet_data_evol
from api.export_alpaca import write_sheet_data_alpaca
from api.export_hhh import write_sheet_data_hhh
import os
import subprocess
import time

def evaluate_model_mt(cfg, i):
    print(f"MT-Bench Evaluating model using {cfg.dataset}_base_{i}.yaml")
    models = [f'{cfg.dataset}_base_{i}_epoch{j+1}' for j in range(3)]
    model_list = ' '.join(models)
    judge_name = f'{cfg.judge_name}'

    cwd = os.getcwd()
    os.chdir("eval/FastChat/fastchat/llm_judge")

    if cfg.parallel_eval is True:
        commands = []
        for i, model in enumerate(models):
            model_path = f'../../../../DC-LLaMA-Factory/{cfg.model_path}/{cfg.dataset}/{model}'
            if i % 2 == 0:
                print(i)
                inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model}" 
            else:
                inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device_2} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model}" 
            commands.append(inf_command)

        max_concurrent = 2  
        processes = []
        # print (commands)
        
        for cmd in commands:
            while len(processes) >= max_concurrent:
                for p in processes:
                    if p.poll() is not None:
                        processes.remove(p)
                time.sleep(1)

            print(f"Starting: {cmd}")
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)

        for p in processes:
            p.wait()

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
            model_path = f'../../../../DC-LLaMA-Factory/{cfg.model_path}/{cfg.dataset}/{model}'
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
    write_sheet_data_mt(i, {cfg.dataset})

def evaluate_model_evol(cfg, i):
    print(f"Evol_Instruct Evaluating model using {cfg.dataset}_base_{i}.yaml")
    models = [f'{cfg.dataset}_base_{i}_epoch{j+1}' for j in range(3)]
    model_list = ' '.join(models)
    judge_name = f'{cfg.judge_name}'

    cwd = os.getcwd()
    os.chdir("eval/FastChat/fastchat/llm_judge")
    if cfg.parallel_eval is True:
        commands = []
        for k, model in enumerate(models):
            model_path = f'../../../../DC-LLaMA-Factory/{cfg.model_path}/{cfg.dataset}/{model}'
            if k % 2 == 0:
                inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model} --bench-name evol_instruct" 
            else:
                inf_command = f"CUDA_VISIBLE_DEVICES={cfg.judge_device_2} PYTHONPATH=../.. python gen_model_answer.py --model-path {model_path} --model-id {model} --bench-name evol_instruct" 
            commands.append(inf_command)

        max_concurrent = 2  
        processes = []
        # print (commands)
        
        for cmd in commands:
            while len(processes) >= max_concurrent:
                for p in processes:
                    if p.poll() is not None:
                        processes.remove(p)
                time.sleep(1)

            print(f"Starting: {cmd}")
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)

        for p in processes:
            p.wait()

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
            model_path = f'../../../../DC-LLaMA-Factory/{cfg.model_path}/{cfg.dataset}/{model}'
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
    write_sheet_data_evol(i, {cfg.dataset})

def evaluate_model_alpaca(cfg, i):
    pass
    # export result to google sheet
    write_sheet_data_alpaca(i, {cfg.dataset})

def evaluate_model_hhh(cfg, i):
    print(f"HHH Evaluating model using {cfg.dataset}_base_{i}.yaml")
    models = [f'{cfg.dataset}_base_{i}_epoch{j+1}' for j in range(3)]
    model_list = ' '.join(models)
    judge_name = f'{cfg.judge_name}'

    cwd = os.getcwd()
    os.chdir("eval/FastChat/fastchat/llm_judge")

    for model in models:
        model_path = f'../../../../DC-LLaMA-Factory/{cfg.model_path}/{cfg.dataset}/{model}'
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
    write_sheet_data_hhh(i, {cfg.dataset})

def evaluate_model(cfg, i):
    if cfg.mt_bench is True:
        evaluate_model_mt(cfg, i)

    if cfg.evol_instruct is True:
        evaluate_model_evol(cfg, i)

    if cfg.alpaca_eval is True:
        evaluate_model_alpaca(cfg, i)

    if cfg.hhh is True:
        evaluate_model_hhh(cfg, i)