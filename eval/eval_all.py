import json
import pdb
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import argparse
#from datasets import dataloader
import base64
import os, glob
from tqdm import tqdm
import cv2, random
import numpy as np
import traceback
import re
# from utils.calculate import  compute_pmi
def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def add_noise(img, mode="gauss"):
    img_array = np.array(img[0])

    if mode == "gauss":
        # 创建高斯噪声
        mean = 0
        sigma = 80  # 控制噪声级别,数值越大噪声越明显
        gauss = np.random.normal(mean, sigma, img_array.shape)
        gauss = gauss.astype(np.int32)

        # 将噪声叠加到图像上
        img_array = img_array + gauss
    elif mode == "pepper":
        # 添加椒盐噪声
        prob = 0.8  # 控制噪声比例
        thres = 1 - prob
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    img_array[i][j] = 0
                elif rdn > thres:
                    img_array[i][j] = 255
    elif mode == "reli":
        # 添加瑞利噪声
        rayleigh_noise = np.random.rayleigh(0.5, size=img_array.shape)
        img_array = img_array + (rayleigh_noise * 255).astype(np.uint8)

    # 将数组转换回 PIL 图像
    noisy_img = Image.fromarray(img_array.astype('uint8'))
    noisy_img.save("noise.png")
    return noisy_img

import torch


def instructblip_inference(model, img, qs, lang, processor, ocr_tokens=None):
    image = Image.open(img).convert("RGB")
    device = "cuda"
    if ocr_tokens is not None:
        prompt = f"OCR tokens: {' '.join(ocr_tokens)}. Question: {qs} Short answer:"
    else:
        prompt = f"Question: {qs} Short answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = inputs.to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return generated_text, None

def llava_inference(model, img, qs, lang, tokenizer, conv, gt=None, mi=False):

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv].copy()

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    # pdb.set_trace()
    prompt = conv.get_prompt()

    images = load_images([img])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    if mi:
        images_wo = [add_noise(images)]
        image_sizes_wo = [x.size for x in images_wo]
        images_tensor_wo = process_images(
            images_wo,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)
        target_ids = (
            tokenizer_image_token(gt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

    with torch.inference_mode():
        if mi:
            outputs_wo = model.generate(
                input_ids,
                # images=images_tensor_wo.unsqueeze(0).half().cuda() if "1.5" in version else images_tensor_wo,
                images=images_tensor_wo,
                image_sizes=image_sizes_wo,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            logits_wo = outputs_wo.scores
            logits_wo = torch.concatenate(logits_wo, dim=0)
            output_ids = outputs_wo.sequences
            response_wo = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print("response wo", response_wo)
        output = model.generate(
            input_ids,
            # images=images_tensor.unsqueeze(0).half().cuda() if "1.5" in version else images_tensor,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        logits = output.scores
        logits = torch.concatenate(logits, dim=0)
        output_ids = output.sequences
        if mi:
            mi, ppl_wo, ppl, e_wo, e = compute_pmi(logits, logits_wo, target_ids)
            mi = {
                "mi": mi,
                "e_wo": ppl_wo,
                "e": ppl,
                "entropy": e,
                "entropy_wo": e_wo,
                "response_wo": response_wo
            }
            ## mi = e - e_wo
        else:
            mi = None
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print("response", outputs)
        return outputs, mi


def monkey_inference(model, img, query, lang, tokenizer, gt=None, mi=False):
    # if lang == "zh":
    #     query = f'<img>{img}</img> 使用一个数字，单词或者句子回答问题 {query} '
    # elif lang == "fr":
    #     query = f'<img>{img}</img> Répondez à la question en utilisant un seul numéro, mot ou phrase {query} '
    # elif lang == "en":
    #     query = f'<img>{img}</img> Answer the question using a single number, word or sentence {query} '
    if mi:
        target_ids = tokenizer(gt, return_tensors="pt", padding="longest").input_ids.cuda()
        input_ids = tokenizer(query, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        pred_wo = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=64,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            return_dict_in_generate=True,
            output_scores=True,
            # output_logits=True
        )
        logits_wo = pred_wo.scores
        logits_wo = torch.concatenate(logits_wo, dim=0)
        output_ids = pred_wo.sequences
        response_wo = tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        print("response wo", response_wo)
    query = f'<img>{img}</img> {query} Answer: '
    input_ids = tokenizer(query, return_tensors='pt', padding='longest')
    attention_mask = input_ids.attention_mask
    input_ids = input_ids.input_ids
    pred = model.generate(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        do_sample=False,
        num_beams=1,
        max_new_tokens=100,
        min_new_tokens=1,
        length_penalty=1,
        num_return_sequences=1,
        output_hidden_states=True,
        use_cache=True,
        pad_token_id=tokenizer.eod_id,
        eos_token_id=tokenizer.eod_id,
        return_dict_in_generate=True,
        output_scores=True
        # output_logits=True
    )
    logits = pred.scores
    logits = torch.concatenate(logits, dim=0)
    output_ids = pred.sequences
    if mi:
        mi, e_wo, e = compute_pmi(logits, logits_wo, target_ids)
        mi = {
            "mi": mi,
            "e_wo": e_wo,
            "e": e
        }
    else:
        mi = None
    response = tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
    return response, mi

def gpt_inference(img_path, question, lang):
    base64_image = encode_image(img_path)
    if lang == "zh":
        prompt = "使用一个数字，单词或者句子回答问题"
    elif lang == "fr":
        prompt = "Répondez à la question en utilisant un seul numéro, mot ou phrase"
    elif lang == "en":
        prompt = "Answer the question using a single number, word or sentence "
    question += "Answer the question using a single number, word or sentence "

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                 }
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

def gemini_inference(model, img_path, question):
    while True:
        response = model.get_response_vision(img_path, question + "Answer the question using a single number, word or sentence in English ", )
        if len(response) > 0:
            return response
        else:
            model = Gemini_Model(key=random.choice(API_LIST))

    # return response

def minicpm_inference(chat_model, img_path, question, lang):
    im_64 = img2base64(img_path)
    # First round chat
    if lang == "zh":
        msgs = [{"role": "user", "content": question + " 使用一个数字，单词或者句子回答问题"}]
    elif lang == "fr":
        msgs = [{"role": "user", "content": question + " Répondez à la question en utilisant un seul numéro, mot ou phrase"}]
    elif lang == "en":
        msgs = [{"role": "user", "content": question + " Answer the question using a single number, word or sentence"}]

    inputs = {"image": im_64, "question": json.dumps(msgs)}
    answer = chat_model.chat(inputs)
    return answer
def cogvlm_inference(model, tokenizer, img_path, question):
    question = f"Question: {question} Answer in short:"
    image = Image.open(img_path).convert("RGB")
    input_by_model = model.build_conversation_input_ids(tokenizer, query=question, history=[], images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).cuda(),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).cuda(),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).cuda(),
        'images': [[input_by_model['images'][0].cuda().to(torch.bfloat16)]] if image is not None else None,
    }
    gen_kwargs = {"max_length": 2048,
                  "do_sample": False}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        outputs = tokenizer.decode(outputs[0])
        outputs = outputs.replace("</s>", "")
        print("\nCog:", outputs)
    return outputs

def mplug_inference(model, img_path, question, tokenizer,image_processor):
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    image = Image.open(img_path).convert('RGB')
    max_edge = max(image.size)  # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    inp = DEFAULT_IMAGE_TOKEN + question + "Answer in a word, number or a short sentence: "
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 0.2
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    outputs = outputs.replace("</s>", "")
    print("outputs", outputs)
    return outputs

def qwenvl_inference(model, img_path, question, lang, tokenizer):
    # 1st dialogue turn
    # if lang == "zh":
    #     question += "\n" + "使用一个数字，单词或者句子回答问题"
    # elif lang == "fr":
    #     question += "\n" + "Répondez à la question en utilisant un seul numéro, mot ou phrase"
    # elif lang == "en":
    #     question = "Answer the question using a single number, word or sentence " + question

    query = tokenizer.from_list_format([
        {'image': img_path},
        # Either a local path or an url
        {'text': question + "Answer: "},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    print("query", query, "response", response)
    return response


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-v1.6-34b") #echo840/Monkey-Chat  echo840/Monkey
    parser.add_argument("--data_path", type=str, default="./data/pageqa")
    parser.add_argument("--human", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default="./result/qasper/ocr/llava-v1.6-34b-2.jsonl")
    parser.add_argument("--conv-mode", type=str, default="chatml_ocr")
    parser.add_argument("--mode", type=str, default="chatml_ocr")
    parser.add_argument("--lang", type=str, default="fr")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--mi", type=bool, default=False)
    parser.add_argument("--ocr", type=bool, default=False)
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')

    args = parser.parse_args()

    if "llava" in args.model_path:
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_PLACEHOLDER,
        )
        from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        checkpoint = args.model_path

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name)
    elif "monkey" in args.model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint = args.model_path
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda', trust_remote_code=True, fp16=True,
                                                     bf16=False).eval()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
    elif "cog" in args.model_path.lower():
        from transformers import AutoModelForCausalLM, LlamaTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
        tokenizer = LlamaTokenizer.from_pretrained("/home/share/models/vicuna-7b-v1.5")
    elif "mplug" in args.model_path:
        from transformers import TextStreamer
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.conversation import conv_templates, SeparatorStyle
        from mplug_owl2.model.builder import load_pretrained_model
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
            KeywordsStoppingCriteria


        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name,
                                                                               load_8bit=False, load_4bit=False,
                                                                               device="cuda")
    elif "Qwen" in args.model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        import torch

        torch.manual_seed(1234)

        # Note: The default behavior now has injection attack prevention off.
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda",
                                                     trust_remote_code=True).eval()
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
    elif "gpt" in args.model_path:
        import openai
        openai.api_base = "https://openkey.cloud/v1"
        openai.api_key = "sk-7MEBBkzTBFQLHSfc4f4fC44e4c054c298a416f81138bA6D6"
        # openai.api_base = "http://127.0.0.1:8000/v1"
        # openai.api_key = "anything"
        # openai.default_headers = {"Authorization": "Bearer anything"}
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        # client = OpenAI(api_key="sk-i9mcaWsHOpu7MmyF60F238Ff2cBe4f66BfF79e9d9b4c4550")


    elif "gemini" in args.model_path:
        from utils.gemini import Gemini_Model
        import itertools
        API_KEY = "AIzaSyCYo6MWJKX4nrV8i36GKVVEVeuYfD3co-s"
        API_KEY4 = 'AIzaSyBFu_DKEssYwlc7MLB_vqQgNOwSJosDAvo'  # yli
        API_KEY5 = 'AIzaSyBbXMPmA8fjG1WN0bPMdMhTLflW_ufJGvY'  # RH

        # API_LIST = [
        #     "AIzaSyC71sLRFmcqw4dWk4G6u4JcfU7oyIdwle0",
        #     "AIzaSyDvgDp0P90FkPrhRu-dH4ckZBgvU9xZA0o",
        #     "AIzaSyDW1zBm_yT6m22_6LiE73iFysZXNyRuSkM",
        #     "AIzaSyBD3PJJ1pGDplOhTjVK3ESNFiC5umisaRk",
        #     "AIzaSyCujTyiBQKNq6M2Z2DeEY5WE7KKMdJTpFw",
        #     "AIzaSyD-LI7LrdqtmUCC4Uh8SLupmSOs-BXvzVM",
        #     "AIzaSyBc1wCA-INgCw-BzcEEw3wNYwG9qvyTP0I",
        #     "AIzaSyAKAFr4BluYrhbgvYjVIdIKQIHPhi49Ugc",
        #     "AIzaSyBDp6XTgF_5oNWxB79eLeg9UaC0xqn23J8",
        #     "AIzaSyD-Kwllyf5KY0VbSlHz22Yu_1qTOH6XU1g",
        #     "AIzaSyCocBiQHc0AWmnK4WvVb3CzjKQ_w_y1kmc",
        #     "AIzaSyDKJMM6OgbH9sVV_nTevWW4SaQe3KSyyv4",
        #     "AIzaSyCi98bty7j9n299249JmN_ZdJWiw-ENs3M",
        #     "AIzaSyCJDJpV_iuvbDuzn0AdVCZ0EDxHm3Z3xJw",
        #     "AIzaSyAI8xqLA9bEH9sVnxAi6oJGcGuRB3YqqMM",
        # ]
        API_LIST = [
            "AIzaSyD-Kwllyf5KY0VbSlHz22Yu_1qTOH6XU1g",
            "AIzaSyCocBiQHc0AWmnK4WvVb3CzjKQ_w_y1kmc",
            "AIzaSyDKJMM6OgbH9sVV_nTevWW4SaQe3KSyyv4",
            "AIzaSyAI8xqLA9bEH9sVnxAi6oJGcGuRB3YqqMM"
        ]
        # API_LIST = [
        #     "AIzaSyCYo6MWJKX4nrV8i36GKVVEVeuYfD3co-s",
        #     'AIzaSyBFu_DKEssYwlc7MLB_vqQgNOwSJosDAvo',
        #     'AIzaSyBbXMPmA8fjG1WN0bPMdMhTLflW_ufJGvY'
        # ]
        models = [Gemini_Model(key=key) for key in API_LIST]
        model_iterator = itertools.cycle(models)
    elif "blip" in args.model_path:
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        import torch
        from PIL import Image
        import requests

        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b").cuda()
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

    # elif "minicpm" in args.model_path:
    #     from transformers import AutoModelForCausalLM, AutoTokenizer
    #     from models.MiniCPM-V.chat import MiniCPMVChat, img2base64
    #     chat_model = MiniCPMVChat(args.model_path)
    mode_dict = {
        "ocrvqa": "ocrvqa",
        "text": "textvqa",
        "doc": "docvqa",
        "chart": "chartvqa",
        "paper": "paper",
        "lunwen": "lunwen"
    }
    if args.mi:
        args.save_path = f'./close_result/{mode_dict[args.mode]}/{args.lang}/{args.model_path.split("/")[-1]}_gauss80.jsonl'
    elif args.ocr:
        args.save_path = f'./close_result/{mode_dict[args.mode]}/{args.lang}/{args.model_path.split("/")[-1]}_ocr.jsonl'
    elif args.human:
        args.save_path = f'./close_result/{mode_dict[args.mode]}/{args.lang}/{args.model_path.split("/")[-1]}_human.jsonl'
    else:
        args.save_path = f'./close_result/{mode_dict[args.mode]}/{args.lang}/{args.model_path.split("/")[-1]}.jsonl'
    print("save path", args.save_path)

    file_mode = "a" if "gemini" in args.model_path else "w"
    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as f:
            gen_length = len(f.readlines())
    else:
        gen_length = 0
    print("gen length", gen_length)
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))
    with open(args.save_path, file_mode) as f:

        if args.mode == "ocr":
            imgs = glob.glob(os.path.join("./data/pageqa/png", "*.png"))
            for img in imgs:
                outputs = llava_inference(model, "")
                f.write(json.dumps({
                    "imgname": img.split('/')[-1].replace(".png", ""),
                    "ocr": outputs,
                }) + "\n")
                print(f"Question: {img} Answer: {outputs}")
        elif args.mode == "chart":
            from dataset.dataloader import ChartvqaLoader
            args.data_path = "./data/ChartQA/test/"
            data_split = "human" if args.human else "augmented"

            if args.lang == "en":
                dataloader = ChartvqaLoader(os.path.join(args.data_path, f"test_{data_split}.json"))
            else:
                dataloader = ChartvqaLoader(os.path.join(args.data_path, f"test_{data_split}_{args.lang}.jsonl"))

            for i, (question, imgname, answers) in enumerate(tqdm(dataloader)):
                img_path = os.path.join(args.data_path, f"png/{imgname}")
                if "llava" in args.model_path:
                    outputs, mi = llava_inference(model, img_path, question, args.lang, tokenizer, args.conv_mode, answers, mi=True)
                elif "monkey" in args.model_path:
                    outputs, mi = monkey_inference(model, img_path, question, args.lang, tokenizer, answers, mi=True)
                elif "Qwen" in args.model_path:
                    outputs, mi = monkey_inference(model, img_path, question, args.lang, tokenizer, answers, mi=True)
                elif "gemini" in args.model_path:
                    if i % 30 != 0:
                        continue
                    model = next(model_iterator)
                    outputs = gemini_inference(model, img_path, question)
                elif "gpt" in args.model_path:
                    if i % 30 != 0:
                        continue
                    outputs = gpt_inference(img_path, question, args.lang)
                elif "blip" in args.model_path:
                    outputs, _ = instructblip_inference(model, img_path, question, args.lang, processor)
                elif "cog" in args.model_path:
                    outputs = cogvlm_inference(model, tokenizer, img_path, question)
                elif "mplug" in args.model_path:
                    outputs = mplug_inference(model, img_path, question, tokenizer, image_processor)
                print(f"Question: {question} Answer: {outputs}")
                f.write(json.dumps({"img_id": imgname,
                                           "prompt": question,
                                           "text": outputs,
                                            "gt": answers,
                                            "mi": mi if args.mi else 0,
                                           "metadata": {}}, ensure_ascii=False) + "\n")

        elif args.mode == "ocrvqa":
            from dataset.dataloader import OcrvqaLoader
            args.data_path = "./data/ocrvqa"
            dataloader = OcrvqaLoader(os.path.join(args.data_path, f"test_{args.lang}.jsonl"))
            for i, (question_id, question, imgname, answer) in enumerate(tqdm(dataloader)):
                if i % 20 != 0:
                    continue
                img_path = os.path.join(args.data_path, f"images/{imgname}.jpg")
                try:
                    if "llava" in args.model_path:
                        outputs, _ = llava_inference(model, img_path, question, args.lang, tokenizer, args.conv_mode)
                    elif "monkey" in args.model_path:
                        outputs, mi = monkey_inference(model, img_path, question, args.lang, tokenizer, answer, mi=True)
                    elif "Qwen" in args.model_path:
                        outputs = qwenvl_inference(model, img_path, question, args.lang, tokenizer)
                    elif "gemini" in args.model_path:

                        model = next(model_iterator)
                        outputs = gemini_inference(model, img_path, question)
                    elif "gpt" in args.model_path:
                        outputs = gpt_inference(img_path, question, args.lang)
                    elif "blip" in args.model_path:
                        outputs, _ = instructblip_inference(model, img_path, question, args.lang, processor)
                    elif "cog" in args.model_path:
                        outputs = cogvlm_inference(model, tokenizer, img_path, question)
                    elif "mplug" in args.model_path:
                        outputs = mplug_inference(model, img_path, question, tokenizer, image_processor)

                except:

                    print(traceback.format_exc())
                    continue

                f.write(json.dumps({
                    "question_id": question_id,
                    "imgname": imgname,
                    "question": question,
                    "gt": answer,
                    "answer": outputs,
                    "mi": mi if args.mi else 0,
                }, ensure_ascii=False) + "\n")
                print(f"Question: {question} Answer: {outputs}")
        elif args.mode == "doc":
            args.data_path = "./data/docvqa/"
            from dataset.dataloader import DocvqaLoader
            if args.lang == "en":
                dataloader = DocvqaLoader(os.path.join(args.data_path, f"qas/val_v1.0_withQT.json"))
            else:
                dataloader = DocvqaLoader(os.path.join(args.data_path, f"qas/val_v1.0_withQT_{args.lang}.jsonl"))
            for i, (question_id, question, image, answers) in enumerate(tqdm(dataloader)):
                img_path = os.path.join(args.data_path, image)
                try:
                    if "llava" in args.model_path:
                        outputs = llava_inference(model, img_path, question, args.lang, tokenizer, args.conv_mode)
                    elif "monkey" in args.model_path:
                        outputs = monkey_inference(model, img_path, question, args.lang, tokenizer)
                    elif "Qwen" in args.model_path:
                        outputs = monkey_inference(model, img_path, question, args.lang, tokenizer)
                    elif "gemini" in args.model_path:
                        if i % 20 != 1:
                            continue
                        model = next(model_iterator)
                        outputs = gemini_inference(model, img_path, question)
                    elif "gpt" in args.model_path:
                        outputs = gpt_inference(client, img_path, question, args.lang)
                    elif "blip" in args.model_path:
                        outputs, _ = instructblip_inference(model, img_path, question, args.lang, processor)
                    elif "cog" in args.model_path:
                        outputs = cogvlm_inference(model, tokenizer, img_path, question)
                    elif "mplug" in args.model_path:
                        outputs = mplug_inference(model, img_path, question, tokenizer, image_processor)
                except:
                    continue


                f.write(json.dumps({
                    "questionId": question_id,
                    "imgname": image,
                    "question": question,
                    "gt": answers,
                    "answer": outputs,
                }, ensure_ascii=False) + "\n")
                print(f"Question: {question} Answer: {outputs}")
        elif args.mode == "text":
            args.data_path = "./data/textvqa"
            from dataset.dataloader import TextvqaLoader
            if args.lang == "en":
                dataloader = TextvqaLoader(os.path.join(args.data_path, f"TextVQA_0.5.1_val.json"))
            else:
                dataloader = TextvqaLoader(os.path.join(args.data_path, f"TextVQA_0.5.1_val_{args.lang}.jsonl"))
            print("data loaded", os.path.join(args.data_path, f"TextVQA_0.5.1_val_{args.lang}.jsonl"))


            for i, (question, imgname, question_id, answers, ocr_tokens) in enumerate(tqdm(dataloader)):
                img_path = os.path.join(args.data_path, f"train_images/{imgname}.jpg")
                if i % 20 != 1:
                    continue
                if "llava" in args.model_path:
                    outputs = llava_inference(model, img_path, question, args.lang, tokenizer, args.conv_mode)
                elif "monkey" in args.model_path:
                    outputs = monkey_inference(model, img_path, question, args.lang, tokenizer)
                elif "Qwen" in args.model_path:
                    outputs = qwenvl_inference(model, img_path, question, args.lang, tokenizer)
                elif "gemini" in args.model_path:

                    model = next(model_iterator)
                    outputs = gemini_inference(model, img_path, question)
                elif "gpt" in args.model_path:
                    outputs = gpt_inference(img_path, question, args.lang)
                elif "blip" in args.model_path:
                    outputs, _ = instructblip_inference(model, img_path, question, args.lang, processor, ocr_tokens)
                elif "cog" in args.model_path:
                    outputs = cogvlm_inference(model, tokenizer, img_path, question)
                elif "mplug" in args.model_path:
                    outputs = mplug_inference(model, img_path, question, tokenizer, image_processor)
                f.write(json.dumps({
                    "question_id": question_id,
                    "imgname": imgname,
                    "question": question,
                    "gt": answers,
                    "answer": outputs,
                }, ensure_ascii=False) + "\n")
                print(f"Question: {question} Answer: {outputs}")
        elif args.mode == "paper":
            from dataset.dataloader import PaperLoader
            if args.lang == "en":
                args.data_path = "./data/pageqa/dev.json"
            else:
                args.data_path = "./data/pageqa/dev_zh.jsonl"
            dataloader = PaperLoader(args.data_path)


            for i, (question, imgname, question_id, answers) in enumerate(tqdm(dataloader)):
                img_path = os.path.join("./data/pageqa/png", f"{imgname}.png")
                if "cog" in args.model_path:
                    outputs = cogvlm_inference(model, tokenizer, img_path, question)
                elif "mplug" in args.model_path:
                    outputs = mplug_inference(model, img_path, question, tokenizer, image_processor)
                elif "Qwen" in args.model_path:
                    outputs = qwenvl_inference(model, img_path, question, args.lang, tokenizer)
                elif "blip" in args.model_path:
                    outputs, _ = instructblip_inference(model, img_path, question, args.lang, processor)
                elif "gpt" in args.model_path:
                    if i % 20 != 3:
                        continue
                    outputs = gpt_inference(img_path, question, args.lang)
                elif "gemini" in args.model_path:
                    if i % 20 != 3:
                        continue
                    model = next(model_iterator)
                    outputs = gemini_inference(model, img_path, question)
                elif "llava" in args.model_path:
                    outputs = llava_inference(model, img_path, question, args.lang, tokenizer, args.conv_mode)
                f.write(json.dumps({
                    "question_id": question_id,
                    "question": question,
                    "answer": outputs,
                    "gt": answers}
                ,ensure_ascii = False
                ) + "\n")
        elif args.mode == "lunwen":
            from dataset.dataloader import LunwenLoader
            data_paths = [f"./data/ch_paper/qas/final-abstractive-{args.lang}-zh.jsonl",
                         f"./data/ch_paper/qas/final-extractive-{args.lang}-zh.jsonl",
                         f"./data/ch_paper/qas/latest-yes-no-{args.lang}-zh.jsonl"]
            types = ['abstractive', 'extractive', 'yesno']
            for data_path, type in zip(data_paths, types):
                dataloader = LunwenLoader(data_path)

                for i, data in enumerate(tqdm(dataloader)):

                    img_path, question, answer = data['img_path'], data['question'], data['answer']
                    img_path = img_path.replace("../", "./")
                    if "cog" in args.model_path:
                        outputs = cogvlm_inference(model, tokenizer, img_path, question)
                    elif "mplug" in args.model_path:
                        outputs = mplug_inference(model, img_path, question, tokenizer, image_processor)
                    elif "Qwen" in args.model_path:
                        outputs = qwenvl_inference(model, img_path, question, args.lang, tokenizer)
                    elif "gpt" in args.model_path:
                        try:
                            if i % 50 != 3:
                                continue
                            outputs = gpt_inference(img_path, question, args.lang)
                        except Exception as e:
                            print(e)
                            continue
                    elif "gemini" in args.model_path:
                        try:
                            if i % 50 != 3:
                                continue
                            model = next(model_iterator)
                            outputs = gemini_inference(model, img_path, question)
                        except:
                            continue
                    elif "blip" in args.model_path:
                        outputs, _ = instructblip_inference(model, img_path, question, args.lang, processor)
                    elif "llava" in args.model_path:
                        outputs = llava_inference(model, img_path, question, args.lang, tokenizer, args.conv_mode)
                    f.write(json.dumps({
                        "img_path": img_path,
                        "question": question,
                        "answer": outputs,
                        "gt": answer,
                        "type": type}
                        , ensure_ascii=False
                    ) + "\n")







