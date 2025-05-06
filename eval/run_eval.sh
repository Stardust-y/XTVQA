#!/bin/bash
#SBATCH -J test                               # 作业名为 test
#SBATCH -o ./log/test-%j.out                           # stdout 重定向到 test.out
#SBATCH -e ./log/test-%j.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 4:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:a100-pcie-40gb:1         # 申请 4 卡 A100 80GB，如果只申请CPU可以删除本行


python eval_xtreme.py \
  --checkpoint qwen \
  --dataset zh.en \
  --eval ../result/qwen/xtreme/zh.en.json