export NAME=$1
python ./eval/eval_qasper.py --predictions close_result/paper/en/gemini-1.5-flash.jsonl \
  --gold ./data/qasper-train-dev-v0.3/qasper-dev-v0.3.json
python ./eval/eval_qasper.py --predictions close_result/paper/zh/gemini-1.5-flash.jsonl \
  --gold ./data/qasper-train-dev-v0.3/qasper-dev-v0.3.json
#python ./eval/eval_qasper.py --predictions results/paper/chpaper/wokl/zh/minicpm/20240814180236/lunwen.json \
#  --gold ./data/qasper-train-dev-v0.3/qasper-dev-v0.3.json
#python ./eval/eval_qasper.py --predictions ./result/paper/zh/${NAME}.jsonl \
#  --gold ./data/qasper-train-dev-v0.3/qasper-dev-v0.3.json
