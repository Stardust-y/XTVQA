import os.path
import pdb
import sys
sys.path.append("..")
sys.path.append("../..")

from dataset import load_dataset
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from utils.render_text import render_text
import openai
import string, collections
import re
openai.api_base = "https://openkey.cloud/v1"
openai.api_key = "sk-iXsrRL5fn1ZM15F54a2aEf41Ff674620BbAdB2A395Bc4a91"

language_dict = {"zh": "Chinese", "en": "English", "fr": "French", "de": "German"}
def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate(args):
    source_lang = args.dataset.split(".")[0]
    dataset = load_dataset("xtreme", f"MLQA.{args.dataset}")
    test_loader = DataLoader(dataset['test'], shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,
                                              trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    with open(f"../result/qwen/xtreme/{args.dataset}.json", "w") as f:
        for item in tqdm(test_loader):
            query = f"Answer the question based on the given context in {language_dict[source_lang]} \n" \
                    f"Context: {item['context'][0]} \n" \
                    f"Question: {item['question'][0]} \n" \
                    f"Answer: "
            response, history = model.chat(tokenizer, query, history=None)
            item['response'] = response
            item['answers']['answer_start'] = item['answers']['answer_start'][0].numpy().tolist()
            print(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def evaluate_vis(args):
    dataset = load_dataset("xtreme", f"MLQA.{args.dataset}")
    test_loader = DataLoader(dataset['test'], shuffle=False)


    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,
                                              trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    image_dir = f"../data/xtreme/{args.dataset}"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    with open(f"../result/qwen/xtreme/{args.dataset}-vis.json", "w") as f:
        for item in tqdm(test_loader):
            id = item['id'][0]
            context = item['context'][0]
            image_path = os.path.join(image_dir, f"{id}.png")
            if not os.path.exists(image_path):
                image = render_text(context)
                image.save(image_path)
            prompt = f'<img>{image_path}</img> Answer the question in {args.dataset.split(".")[0]} Question:{item["question"]} '
            response = model.chat(tokenizer, query=prompt, history=None)
            item['response'] = response
            item['answers']['answer_start'] = item['answers']['answer_start'][0].numpy().tolist()
            print(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def evaluate_gpt(args):
    dataset = load_dataset("xtreme", f"MLQA.{args.dataset}")
    test_loader = DataLoader(dataset['test'], shuffle=False)
    save_dir = f"../result/gpt3.5/xtreme/{args.dataset}.jsonl"
    with open(save_dir, "r") as f:
        length = len(f.readlines())
    with open(save_dir, "a") as f:
        for i, item in enumerate(tqdm(test_loader)):
            if i < length:
                continue
            query = f"Answer the question based on the given context in {language_dict[args.dataset.split('.')[0]]} \n" \
                    f"Context: {item['context'][0]} \n" \
                    f"Question: {item['question'][0]} \n" \
                    f"Answer: "
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{'role': 'system', 'content': query}],
                        temperature=0.2,
                        stop='\n\n',
                    )
                    item['response'] = response.choices[0].message.content
                    item['answers']['answer_start'] = item['answers']['answer_start'][0].numpy().tolist()
                    print(item)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    break
                except:
                    continue

def calc_f1(file_path):
    scores = []
    with open(file_path, "r") as f:
        for line in f:
            result = eval(line)
            f1 = compute_f1(result['response'], result['answers']['text'][0][0])
            scores.append(f1)

    print("Accuracy: ", sum(scores) / len(scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a series of point clouds as an animation.")
    parser.add_argument("--dataset", type=str, default='zh.en')
    parser.add_argument("--savedir", type=str, default="../data/lunwen/qa")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/Qwen-VL")
    parser.add_argument("--eval", type=str)
    args = parser.parse_args()
    if "VL" in args.checkpoint:
        #evaluate_vis(args)
        calc_f1(f"../result/qwen/xtreme/{args.dataset}-vis.json")
    elif "gpt" in args.checkpoint:
        # evaluate_gpt(args)
        calc_f1(f"../result/gpt3.5/xtreme/{args.dataset}.jsonl")
    else:
        #evaluate(args)
        calc_f1(f"../result/qwen/xtreme/{args.dataset}.json")
