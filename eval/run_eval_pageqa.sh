export MODE=$1
python eval_pageqa.py \
 --predictions ../result/minicpm/ch_paper/origin/en-zh-$MODE.jsonl \
 --lang zh

python eval_pageqa.py \
 --predictions ../result/minicpm/ch_paper/origin/zh-zh-$MODE.jsonl \
 --lang zh

python eval_pageqa.py \
 --predictions ../result/minicpm/ch_paper/origin/en-en-$MODE.jsonl \
 --lang en

python eval_pageqa.py \
 --predictions ../result/minicpm/ch_paper/origin/zh-en-$MODE.jsonl \
 --lang en
