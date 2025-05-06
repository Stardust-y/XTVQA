#!/bin/bash
#SBATCH -J test                               # 作业名为 test
#SBATCH -o ./log/llava/test-%j.out                           # stdout 重定向到 test.out
#SBATCH -e ./log/llava/test-%j.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 10:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export MODEL=$1
python ./eval/llava/eval_qasper.py --conv-mode llava_v1 \
  --model_path ./checkpoints/${MODEL} \
  --lang en \
  --save_path ./results/paper/enpaper/dev/en/${MODEL}.jsonl
python ./eval/llava/eval_qasper.py --conv-mode llava_v1 \
  --model_path ./checkpoints/${MODEL} \
  --lang zh \
  --save_path ./results/paper/enpaper/dev/zh/${MODEL}.jsonl