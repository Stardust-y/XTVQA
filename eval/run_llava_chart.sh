#!/bin/bash
#SBATCH -J train_story-dalle
#SBATCH -o ./log/chart/h5-%j.out
#SBATCH -e ./log/chart/h5-%j.err
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH -N 1
#SBATCH -t 16:00:00

python eval/llava/eval_chartvqa.py \
    --model-path ./checkpoints/monkey \
    --question-file ./data/ChartQA/test/test_human_zh.jsonl \
    --image-folder ./data/ChartQA/test/png \
    --table-folder ./data/ChartQA/test/tables \
    --answers-file ./result/ChartQA/zh/moonkey.jsonl \
    --temperature 0 \
    --conv-mode chatml_direct \
    --lang zh

python eval/llava/eval_chartvqa.py \
    --model-path ./checkpoints/monkey \
    --question-file ./data/ChartQA/test/test_human_fr.jsonl \
    --image-folder ./data/ChartQA/test/png \
    --table-folder ./data/ChartQA/test/tables \
    --answers-file ./result/ChartQA/fr/monkey.jsonl \
    --temperature 0 \
    --conv-mode chatml_direct \
    --lang fr

#python ./eval/llava/acc_chartvqa.py \
#    --annotation-file ./data/ChartQA/test/test_human.json \
#    --result-file ./data/ChartQA/answers/human_13b.jsonl

# --model-path ./llava/checkpoints/llava-v1.5-13b \
