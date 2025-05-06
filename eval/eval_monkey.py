import json
import sys
sys.path.append("..")
sys.path.append("../..")
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from dataset.dataloader import PaperLoader, LunwenLoader
import os, glob

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../checkpoints/monkey") #echo840/Monkey-Chat  echo840/Monkey
    parser.add_argument("--data_path", type=str, default="../data/pageqa")
    parser.add_argument("--save_path", type=str, default="../results/paper/enpaper/en/monkey.jsonl")
    parser.add_argument("--mode", type=str, default="paper")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()
    checkpoint = args.model_path
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda', trust_remote_code=True, fp16=True, bf16=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    if args.mode == "paper":
        if args.lang == "en":
            dataloader = PaperLoader(os.path.join(args.data_path, "dev.json"))
        else:
            dataloader = PaperLoader(os.path.join(args.data_path, "dev_zh_en.jsonl"))
        with open(args.save_path, "w") as f:
            for question, imgname, question_id, answers in dataloader:
                img_path = os.path.join(args.data_path, f"png/{imgname}.png")
                query = f'<img>{img_path}</img> {question} Answer: '  # VQA

                input_ids = tokenizer(query, return_tensors='pt', padding='longest')
                attention_mask = input_ids.attention_mask
                input_ids = input_ids.input_ids

                pred = model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                )
                response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
                f.write(json.dumps({
                    "question_id": question_id,
                    "question": question,
                    "answer": response
                }) + '\n')
                print(f"Question: {question} Answer: {response}")
    elif args.mode == "lunwen":
        dataloader = LunwenLoader(args.data_path)
        with open(args.save_path, "w") as f:
            for data in dataloader:
                img_path, question, answer = data['img_path'], data['question'], data['answer']
                query = f'<img>{img_path}</img> {question} Answer: '  # VQA

                input_ids = tokenizer(query, return_tensors='pt', padding='longest')
                attention_mask = input_ids.attention_mask
                input_ids = input_ids.input_ids

                pred = model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                )
                response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
                f.write(json.dumps({
                    "gt_answer": answer,
                    "question": question,
                    "answer": response
                }, ensure_ascii=False) + '\n')
                print(f"Question: {question} Answer: {response}")

