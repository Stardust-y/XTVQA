#!/bin/bash
#SBATCH -J train_story-dalle
#SBATCH -o ./log/chart/h5-%j.out
#SBATCH -e ./log/chart/h5-%j.err
#SBATCH -p compute
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH -N 1
#SBATCH -t 16:00:00

python eval/llava/eval_snli.py \
    --model-path ./checkpoints/llava-v1.5-13b \
    --image-folder ./data/SNLI-VE/data/images \
    --data-file ./data/SNLI-VE/data/snli_ve_dev.jsonl \
    --answers-file ./result/SNLI/answers/llava-v1.5-13b-attn.jsonl \
    --temperature 0 \
    --conv-mode chatml_direct

python ./eval/llava/acc_chartvqa.py \
    --annotation-file ./data/ChartQA/test/test_human.json \
    --result-file ./data/ChartQA/answers/human_13b.jsonl