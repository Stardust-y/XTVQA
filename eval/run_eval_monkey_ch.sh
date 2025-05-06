#!/bin/bash
#SBATCH -J test                               # 作业名为 test
#SBATCH -o ./log/test-%j.out                           # stdout 重定向到 test.out
#SBATCH -e ./log/test-%j.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 10:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-pcie-40gb:1

export MODEL=$1

if [ $MODEL = "monkey" ]
then
  for lang in "en" "zh"
  do
    for mode in "extractive" "abstractive"
    do
      python ./eval_monkey.py --model_path ../checkpoints/monkey \
         --data_path ../data/ch_paper/qas/final-${mode}-${lang}-zh.jsonl \
         --save_path ../results/paper/chpaper/${lang}/monkey-${mode}.jsonl \
         --mode lunwen
    done
    python ./eval_monkey.py --model_path ../checkpoints/monkey \
         --data_path ../data/ch_paper/qas/latest-yes-no-${lang}-zh.jsonl \
         --save_path ../results/paper/chpaper/${lang}/monkey-yes-no.jsonl \
         --mode lunwen
  done
elif [ $MODEL = "qwen" ]
then
  for lang in "en" "zh"
  do
    for mode in "extractive" "abstractive"
    do
      python ./eval_monkey.py --model_path ../checkpoints/Qwen/Qwen-VL-Chat \
         --data_path ../data/ch_paper/qas/final-${mode}-${lang}-zh.jsonl \
         --save_path ../results/paper/chpaper/${lang}/qwen-${mode}.jsonl \
         --mode lunwen
    done
    python ./eval_monkey.py --model_path ../checkpoints/Qwen/Qwen-VL-Chat \
         --data_path ../data/ch_paper/qas/latest-yes-no-${lang}-zh.jsonl \
         --save_path ../results/paper/chpaper/${lang}/qwen-yes-no.jsonl \
         --mode lunwen
  done
fi

