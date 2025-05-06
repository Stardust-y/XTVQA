#!/bin/bash
#SBATCH -J test                               # 作业名为 test
#SBATCH -o ./log/test-%j.out                           # stdout 重定向到 test.out
#SBATCH -e ./log/test-%j.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 10:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1

export MODEL=$1

if [ $MODEL = "monkey" ]
then
  python ./eval_monkey.py --model_path ../checkpoints/monkey \
   --data_path ../data/pageqa \
   --save_path ../results/paper/enpaper/dev/en/monkey.jsonl \
   --lang en

  python ./eval_monkey.py --model_path ../checkpoints/monkey \
   --data_path ../data/pageqa \
   --save_path ../results/paper/enpaper/dev/zh/monkey.jsonl \
   --lang zh
elif [ $MODEL = "qwen" ]
then
  python ./eval_monkey.py --model_path ../checkpoints/Qwen/Qwen-VL-Chat \
   --data_path ../data/pageqa \
   --save_path ../results/paper/enpaper/dev/en/qwenvl.jsonl \
   --lang en

  python ./eval_monkey.py --model_path ../checkpoints/Qwen/Qwen-VL-Chat \
   --data_path ../data/pageqa \
   --save_path ../results/paper/enpaper/dev/zh/qwenvl.jsonl \
   --lang zh
fi

