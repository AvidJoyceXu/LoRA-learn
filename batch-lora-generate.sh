for lora_dim in 2 4 8 16
do
    lora_load_path="results/lora-${lora_dim}/lora.pt"
    output_dir_name="lora-${lora_dim}-eval"

    CUDA_VISIBLE_DEVICES=1 python generate.py \
        --model_name_or_path gpt2 \
        --max_length 512 \
        --trust_remote_code True \
        --use_lora True \
        --lora_dim $lora_dim \
        --lora_scaling 32 \
        --lora_module_name h. \
        --lora_load_path $lora_load_path \
        --seed 42 \
        --use_cuda True \
        --output_dir_name $output_dir_name
done
