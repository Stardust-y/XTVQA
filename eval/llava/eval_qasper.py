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
from dataset.dataloader import PaperLoader, LunwenLoader
import os
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
    parser.add_argument("--save_path", type=str, default="./result/qasper/llava-v1.5-13b.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--mode", type=str, default="lunwen")
    args = parser.parse_args()
    if args.mode == "lunwen":
        dataloader = LunwenLoader(args.data_path)
    else:
        if args.lang == "en":
            dataloader = PaperLoader(os.path.join(args.data_path, "dev.json"))
        else:
            dataloader = PaperLoader(os.path.join(args.data_path, "dev_zh_en.jsonl"))

    if "llava" in args.model_path:
        checkpoint = args.model_path

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name)

        if not os.path.exists(os.path.dirname(args.save_path)):
            os.makedirs(os.path.dirname(args.save_path))
        with open(args.save_path, "w") as f:
            if args.mode == "lunwen":
                for data in dataloader:
                    img_path, question, answer = data['img_path'], data['question'], data['answer']
                    img_path = img_path.replace("../", "./")
                    if model.config.mm_use_im_start_end:
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    print("conv", conv)
                    # pdb.set_trace()
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                                      return_tensors='pt').unsqueeze(
                        0).cuda()
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = process_images([image], image_processor, model.config)[0]
                    print("img tensor", image_tensor.shape)
                    # print("input_ids", input_ids)
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
                        "question": question,
                        "answer": answer,
                        "gt_answer": outputs,
                    }, ensure_ascii=False) + "\n")
                    print(f"Question: {question} Answer: {outputs}")
            else:
                for question, imgname, question_id, answers in dataloader:
                    img_path = f"{args.data_path}/png/{imgname}.png"
                    if model.config.mm_use_im_start_end:
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    print("conv", conv)
                    # pdb.set_trace()
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                        0).cuda()
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = process_images([image], image_processor, model.config)[0]
                    print("img tensor", image_tensor.shape)
                    # print("input_ids", input_ids)
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
                        "question_id": question_id,
                        "predicted_answer": outputs,
                    }, ensure_ascii=False) + "\n")
                    print(f"Question: {question} Answer: {outputs}")



