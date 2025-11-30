CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model gpt-4o-2024-11-20 --model-list llama3-8b-instruct --parallel 2
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../.. python gen_judgment_noenter.py --judge-model gpt-4o-2024-11-20 --model-list llama3-8b-instruct  --parallel 2 --mode pairwise-baseline
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../.. python show_result.py --judge-model gpt-4o-2024-11-20 --model-list llama3-8b-instruct
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../.. python show_result.py --judge-model gpt-4o-2024-11-20 --model-list llama3-8b-instruct --mode pairwise-baseline