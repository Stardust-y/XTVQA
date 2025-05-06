export MODEL=qwen
export NAME=llava-v1.6-34b
python ./eval/eval_pageqa.py --predictions result/lunwen/en/llava-v1.5-13b-trans.jsonl \
  --language mix
python ./eval/eval_pageqa.py --predictions result/lunwen/en/llava-v1.6-34b-trans.jsonl \
  --language mix
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/full/en/minicpm/20240814170301/lunwen-trans.jsonl \
#  --language yesno
#python ./eval/eval_pageqa.py --predictions ./result/lunwen/zh/${NAME}.jsonl \
#  --language mix

#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/en/${MODEL}-abstractive.jsonl \
#  --language zh
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/en/${MODEL}-extractive.jsonl \
#  --language zh
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/en/${MODEL}-yes-no.jsonl \
#  --language yesno
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/zh/minicpm/20240803022751/lunwen.json \
#  --language zh
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/wokl/en/minicpm/20240813161800/lunwen.json \
# --language zh
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/wokl/en/minicpm/20240813181343/lunwen.json \
# --language zh
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/en/minicpm/20240813195649/lunwen.json \
# --language yesno
#
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/wokl/zh/minicpm/20240813202906/lunwen.json \
# --language zh
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/wokl/zh/minicpm/20240813225802/lunwen.json \
# --language zh
#python ./eval/eval_pageqa.py --predictions results/paper/chpaper/zh/minicpm/20240814011900/lunwen.json \
# --language yesno


