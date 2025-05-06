export NAME=instructblip
export MODE=$1

if [ $MODE = "ocr" ]
then
  python ./eval/make_score.py --result-file ./close_result/ocrvqa/en/gemini-1.5-pro-flash_human.jsonl --mode ocr
  python ./eval/make_score.py --result-file ./close_result/ocrvqa/zh/gemini-1.5-pro-flash_human.jsonl --mode ocr
  python ./eval/make_score.py --result-file ./close_result/ocrvqa/fr/gemini-1.5-pro-flash_human.jsonl --mode ocr
elif [ $MODE = "chart" ]
then
#  python ./eval/make_score.py --result-file ./result/chartvqa/zh/minicpm/20240815125315/chartVQA.json --mode chart
#  python ./eval/make_score.py --result-file ./result/chartvqa/zh/${NAME}.jsonl --mode chart
#  python ./eval/make_score.py --result-file ./result/chartvqa/fr/${NAME}.jsonl --mode chart
  python ./eval/make_score.py --result-file ./close_result/chartvqa/en/gemini-1.5-pro-flash.jsonl --mode chart
  python ./eval/make_score.py --result-file ./close_result/chartvqa/zh/gemini-1.5-pro-flash.jsonl --mode chart
  python ./eval/make_score.py --result-file ./close_result/chartvqa/fr/gemini-1.5-pro-flash.jsonl --mode chart

elif [ $MODE = "text" ]
then
  python ./models/LLaVA/llava/eval/eval_textvqa.py --annotation-file ./data/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./close_result/textvqa/en/gemini-1.5-pro-exp_human.jsonl --mode single
  python ./models/LLaVA/llava/eval/eval_textvqa.py --annotation-file ./data/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./close_result/textvqa/zh/gemini-1.5-pro-exp_human.jsonl --mode single
  python ./models/LLaVA/llava/eval/eval_textvqa.py --annotation-file ./data/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./close_result/textvqa/fr/gemini-1.5-pro-exp_human.jsonl --mode single
#  python ./models/LLaVA/llava/eval/eval_textvqa.py --annotation-file ./data/textvqa/TextVQA_0.5.1_val.json \
#    --result-file ./result/textvqa/zh/${NAME}.jsonl --mode single
elif [ $MODE = "doc" ]
then
  python ./models/Qwen-VL/eval_mm/infographicsvqa_eval.py -g ./data/docvqa/qas/val_v1.0_withQT.json \
    --s ./result/gemini/result/docvqa/en/gemini-1.5-flash_gauss80.jsonl
  python ./models/Qwen-VL/eval_mm/infographicsvqa_eval.py -g ./data/docvqa/qas/val_v1.0_withQT.json \
    --s ./result/docvqa/fr/gemini-1.5-flash_human.jsonl
  python ./models/Qwen-VL/eval_mm/infographicsvqa_eval.py -g ./data/docvqa/qas/val_v1.0_withQT.json \
    --s ./result/gemini/result/docvqa/zh/gemini-1.5-flash_gauss80.jsonl
#  python ./models/LLaVA/llava/eval/eval_textvqa.py --annotation-file ./data/textvqa/TextVQA_0.5.1_val.json \
#    --result-file ./result/textvqa/en/${NAME}.jsonl
#  python ./models/LLaVA/llava/eval/eval_textvqa.py --annotation-file ./data/textvqa/TextVQA_0.5.1_val.json \
#    --result-file ./result/textvqa/zh/${NAME}.jsonl
#  python ./models/LLaVA/llava/eval/eval_textvqa.py --annotation-file ./data/textvqa/TextVQA_0.5.1_val.json \
#    --result-file ./result/textvqa/fr/${NAME}.jsonl
fi