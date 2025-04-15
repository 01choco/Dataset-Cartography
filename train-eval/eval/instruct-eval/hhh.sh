#p hhh.py main --model_name openai --model_path VisualQuestionAnswering --use_azure


model=(
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_highavg_dpo1"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_highavg_dpo2"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_highavg_dpo3"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_highvar_dpo1"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_highvar_dpo2"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_highvar_dpo3"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_lowavg_dpo1"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_lowavg_dpo2"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_lowavg_dpo3"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_rand42_dpo1"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_rand42_dpo2"
    "/data/dataset_cartography/DC-LLaMA-Factory/models/llama2-13b/dpo/hyperparameter/llama2-13b_hp_rand42_dpo3"


) #carto
for model in "${model[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python hhh.py main --model_name llama --model_path ${model} --load_8bit
done


