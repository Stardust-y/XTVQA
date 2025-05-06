import json
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from dataset.dataloader import DataLoader
import os, glob

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../checkpoints/monkey") #echo840/Monkey-Chat  echo840/Monkey
    parser.add_argument("--data_path", type=str, default="../data/pageqa")
    parser.add_argument("--save_path", type=str, default="../results/monkey.jsonl")
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--mode", type=str, default="ocr")
    args = parser.parse_args()
    checkpoint = args.model_path
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda', trust_remote_code=True, fp16=True, bf16=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id


    with open(args.save_path, "w") as f:
        if args.mode == "ocr":
            imgs = glob.glob(os.path.join("../data/pageqa/png", "*.png"))
            for img in imgs:
                query = f'<img>{img}</img> Identify all the text in the image: '  # VQA

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
                    "question": img,
                    "answer": response
                }, indent=4))
                print(f"Question: {img} Answer: {response}")
        elif args.mode == "chart":
            from dataset.dataloader import ChartvqaLoader
            dataloader = ChartvqaLoader(os.path.join(args.data_path, f"test_human_{args.lang}.jsonl"))
            for question, imgname, answers in dataloader:
                img_path = os.path.join(args.data_path, f"png/{imgname}")
                if args.lang == "zh":
                    query = f'<img>{img_path}</img> {question} 使用一个数字，单词或者句子回答问题r: ' #VQA
                elif args.lang == "fr":
                    query = f'<img>{img_path}</img> {question} Répondez à la question en utilisant un seul numéro, mot ou phrase: '  # VQA

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
                    "question": question,
                    "answer": response
                }, indent=4))
                print(f"Question: {question} Answer: {response}")
        elif args.mode == "st":
            from dataset.dataloader import stvqaLoader
            dataloader = stvqaLoader(os.path.join(args.data_path, f"qas/val_v1.0_withQT_{args.lang}.jsonl"))
            for question_id, question, imgname, answers in dataloader:
                img_path = os.path.join(args.data_path, f"{imgname}")
                if args.lang == "zh":
                    query = f'<img>{img_path}</img> {question} 使用一个数字，单词或者句子回答问题r: ' #VQA
                elif args.lang == "fr":
                    query = f'<img>{img_path}</img> {question} Répondez à la question en utilisant un seul numéro, mot ou phrase: '  # VQA

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
                }, indent=4))
                print(f"Question: {question} Answer: {response}")
