#!/bin/bash
#SBATCH -J $1-$2                               # 作业名为 test
#SBATCH -o ./log/eval/test-%j.out                           # stdout 重定向到 test.out
#SBATCH -e ./log/eval/test-%j.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 10:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-sxm4-80gb:1

#python ./eval/eval_all.py --data_path "./data/ChartQA/test" \
#  --model_path "./checkpoints/monkey" \
#  --lang "en" \
#  --conv-mode "llava_v1" \
#  --mode chart \
#  --save_path "./result/ChartQA/en/monkey.jsonl"
export DATASET=$2
if [ $1 = "monkey" ]
then
  python ./eval/eval_all.py --data_path "./data/ocrvqa/" \
    --model_path "./checkpoints/monkey" \
    --lang "en" \
    --conv-mode "llava_v1" \
    --mode $DATASET \
    --save_path "./result/ocrvqa/en/monkey_mi.jsonl" \
    --mi True

  python ./eval/eval_all.py --data_path "./data/ocrvqa/" \
    --model_path "./checkpoints/monkey" \
    --lang "fr" \
    --conv-mode "llava_v1" \
    --mode $DATASET \
    --save_path "./result/ocrvqa/fr/monkey_mi.jsonl" \
    --mi True

  python ./eval/eval_all.py --data_path "./data/ocrvqa/" \
    --model_path "./checkpoints/monkey" \
    --lang "zh" \
    --conv-mode "llava_v1" \
    --mode $DATASET \
    --save_path "./result/ocrvqa/zh/monkey_mi.jsonl" \
    --mi True
elif [ $1 = "qwen" ]
then
  python ./eval/eval_all.py --data_path "./data/textvqa/" \
    --model_path "./checkpoints/Qwen/Qwen-VL-Chat" \
    --lang "en" \
    --conv-mode "llava_v1" \
    --mode $DATASET \
    --save_path "./result/textvqa/en/qwen.jsonl"

  python ./eval/eval_all.py --data_path "./data/textvqa/" \
    --model_path "./checkpoints/Qwen/Qwen-VL-Chat" \
    --lang "zh" \
    --conv-mode "llava_v1" \
    --mode $DATASET \
    --save_path "./result/textvqa/zh/qwen.jsonl"
elif [ $1 = "llava" ]
then
  for lang in "zh"
  do
    python ./eval/eval_all.py --data_path "./data/textvqa/" \
      --model_path "./checkpoints/llava-v1.6-34b" \
      --lang ${lang} \
      --conv-mode "llava_v1" \
      --mode $DATASET \
      --save_path "./result/textvqa/en/llava-v1.5-13b.jsonl"
  done
elif [ $1 = "blip" ]
then
  for lang in "zh"
  do
    python ./eval/eval_all.py --data_path "./data/textvqa/" \
      --model_path "./checkpoints/instructblip" \
      --lang ${lang} \
      --mode $DATASET \
      --ocr True
  done
elif [ $1 = "cog" ]
then
  for lang in "en" "zh"
  do
    python ./eval/eval_all.py  \
      --model_path "./checkpoints/cogvlm-chat-hf" \
      --lang ${lang} \
      --mode $DATASET
  done
elif [ $1 = "mplug" ]
then
  for lang in "zh"
  do
    python ./eval/eval_all.py  \
      --model_path "./checkpoints/mplug-owl2" \
      --lang ${lang} \
      --mode $DATASET
  done
elif [ $1 = "gemini" ]
then
  for lang in "en" "zh"
  do
    python ./eval/eval_all.py --data_path "./data/textvqa" --mode ${DATASET} --lang ${lang} \
      --model_path gemini-1.5-flash
  done
elif [ $1 = "gpt" ]
then
  for lang in "en"
  do
    python ./eval/eval_all.py --data_path "./data/textvqa" --mode ${DATASET} --lang ${lang} \
      --model_path gpt-4o-v3
  done
fi
