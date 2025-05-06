import json
import pdb
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
#from datasets import dataloader
from dataset.dataloader import DataLoader
import os, glob
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoModelForCausalLM, LlamaTokenizer

from PIL import Image

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-v1.6-34b") #echo840/Monkey-Chat  echo840/Monkey
    parser.add_argument("--data_path", type=str, default="./data/pageqa")
    parser.add_argument("--save_path", type=str, default="./result/qasper/ocr/llava-v1.6-34b-2.jsonl")
    parser.add_argument("--conv-mode", type=str, default="chatml_ocr")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    dataloader = DataLoader(os.path.join(args.data_path, "dev.json"))

    if "llava" in args.model_path:
        checkpoint = args.model_path

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name)

    with open(args.save_path, "w") as f:
        imgs = glob.glob(os.path.join("./data/pageqa/png", "*.png"))
        for img in imgs:

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n'

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            print("conv", conv)
            # pdb.set_trace()
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()
            image = Image.open(img).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0]
            print("img tensor", image_tensor.shape)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            f.write(json.dumps({
                "imgname": img.split('/')[-1].replace(".png", ""),
                "ocr": outputs,
            }) + "\n")
            print(f"Question: {img} Answer: {outputs}")



