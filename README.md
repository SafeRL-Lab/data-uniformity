


## Data Uniformity Improves Training Efficiency and More, with a Convergence Framework Beyond the NTK Regime





<h2 id="usage">Usage</h2>

- Setup. Install the conda environment:
```bash
conda create -n llamax python=3.10
conda activate llamax
git clone https://github.com/AetherCortex/Llama-X.git
cd Llama-X/src
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install transformers==4.31.0
cd ../..
pip install -r requirements.txt
```

- Training data example (e.g., [TeaMs-RL](https://github.com/SafeRL-Lab/TeaMs-RL)):
```bash
dataset/teams_rl/full_dataset_9k.json
```

- Convert LLaMA checkpoint to HuggingFace format:
```bash
cd src
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/llama-7B/ \
    --model_size 7B \
    --output_dir /path/to/llama-7B/hf
```

- Train LLaMA-7B on DeepSpeed Zero-3:
```bash
deepspeed train.py \
    --model_name_or_path /path/to/llama-7B/hf \
    --data_path /path/to/example_data.json \
    --output_dir /path/to/llama-7B/hf/ft \
    --num_train_epochs 3 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
```
- Train LLaMA-7B on DeepSpeed Zero-3 with Multi-nodes
```bash
deepspeed --num_gpus num_of_gpus_in_each_node \
    --num_nodes num_of_nodes \
    --master_addr ip_address_of_main_node \
    --master_port 34545 \
    --hostfile configs/hostfile \
    train.py \
    --model_name_or_path /path/to/llama-7B/hf \
    --data_path /path/to/example_data.json \
    --output_dir /path/to/llama-7B/hf/ft \
    --num_train_epochs 3 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
```

## Citation
If you find the repository useful, please cite the study
``` Bash
@article{wang2025uniformity,
  title={Data Uniformity Improves Training Efficiency and More, with a Convergence Framework Beyond the NTK Regime},
  author={Wang, Yuqing and Gu, Shangding},
  journal={Github},
  year={2025}
}
```


## Thanks For

This project has been inspired by multiple open source projects:

[Llama-X](https://github.com/AetherCortex/Llama-X)

[Meta AI LLaMA](https://arxiv.org/abs/2302.13971v1)

[Huggingface Transformers Llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)

[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)






