python ./eval/llava/eval_mathvqa.py --conv-mode llava_v1 \
  --model-path ./checkpoints/llava-v1.6-34b \
  --question-file ./data/pazhou/test.jsonl \
  --answers-file ./results/pazhou/llava-v1.6-34b.jsonl \
  --image-folder ./data/pazhou \
  --lang ch