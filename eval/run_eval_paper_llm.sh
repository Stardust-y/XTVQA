export MODEL=$1
if [ $MODEL = "llama" ]
then
  for lang in "zh" "en"
  do
    python ./eval/eval_paper_llm.py --model_path /home/share/models/Meta-Llama-3.1-8B-Instruct \
      --save_path ./results/paper/llm/enpaper/${lang}/${MODEL}.jsonl \
      --lang ${lang}
  done
elif [ $MODEL = "vicuna" ]
then
  for lang in "zh" "en"
  do
    python ./eval/llm/eval_paper_llm.py --model_path /home/share/models/vicuna-13b-v1.5-16k \
      --save_path ./results/paper/llm/enpaper/${lang}/${MODEL}.jsonl \
      --lang ${lang}
  done
elif [ $MODEL = "minicpm" ]
then
  for lang in "zh" "en"
  do
    python ./eval/llm/eval_paper_llm.py --model_path /home/share/models/MiniCPM-2B-sft-bf16 \
      --save_path ./results/paper/llm/enpaper/${lang}/${MODEL}.jsonl \
      --lang ${lang}
  done
fi