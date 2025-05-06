import torch
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import argparse
from dataset.dataloader import DataLoader
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--model_path", type=str, default="./checkpoints/cog-agent-hf") #echo840/Monkey-Chat  echo840/Monkey
parser.add_argument("--data_path", type=str, default="./data/pageqa")
parser.add_argument("--save_path", type=str, default="./result/cogagent/qasper_zh.jsonl")

args = parser.parse_args()
dataloader = DataLoader(os.path.join(args.data_path, "dev_back.jsonl"))
MODEL_PATH = args.model_path
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=args.quant is not None,
        trust_remote_code=True
    ).to(DEVICE).eval()

with open(args.save_path, "w") as f:
    for question, imgname, question_id, answers in dataloader:
        img_path = f"{args.data_path}/png/{imgname}.png"

        image = Image.open(img_path).convert('RGB')
        history = []

        query = question + "Answer in one short sentence: "
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history,
                                                            images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048,
                      "temperature": 0.9,
                      "do_sample": False}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            print("\nCog:", response)
        f.write(json.dumps({
            "question_id": question_id,
            "predicted_answer": response,
        }) + "\n")
        print(f"Question: {question} Answer: {response}")

