import argparse
import pdb

import torch
import os
import json
from tqdm import tqdm
import shortuuid
import cv2
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers.generation.configuration_utils import GenerationConfig
from PIL import Image
import math
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import functional as F

PROMPT = "Given a picture and a hypothesis, you need to judge the relationship between the picture and the text, " \
         "There're three options: Entailment, Neutral or Contradiction." \
         "Here's the hypothesis: <hypothesis> " \
         "Answer with only one word: "
class CustomDataset(Dataset):
    def __init__(self, datafile, image_folder, tokenizer, image_processor, model_config):
        with open(datafile, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        data = self.data[index]

        qs = data['sentence2']
        qs = PROMPT.replace("<hypothesis>", qs)
        image_file = data['Flickr30K_ID'] + ".jpg"
        print(image_file)
        pair_id = data['pairID']

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        img_dir = os.path.join(self.image_folder, image_file)
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        img_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, prompt, img_size, pair_id, img_dir

    def __len__(self):
        return len(self.data)

def create_data_loader(data_file, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(data_file, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(args.data_file, args.image_folder, tokenizer, image_processor, model.config)

    for i, (input_ids, image_tensor, prompt, img_size, pair_id, image_file) in enumerate(tqdm(data_loader)):
        if "1.6-34b" in args.model_path:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        # generation_config = {
        #     "output_attentions": True,
        # }

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[img_size],
                output_attentions=True,
                return_dict=True
            )

            attn = outputs['attentions'][0]

            print(attn.shape) # q为query的序列长(这里设为16)，k为key的序列长（这里表示图像feature的patch数192=24*8）
            attn_map = attn[0][15, :, :].detach().cpu().numpy()
            print(attn_map)

            img = cv2.imread(image_file[0])

            attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))

            # 创建热力图颜色映射
            heat_map = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
            heat_map = cv2.resize(heat_map, (img.shape[1], img.shape[0]))

            # 将热力图和原始图像融合
            img_with_heatmap = cv2.addWeighted(img, 0.5, heat_map, 0.5, 0)

            # 保存结果图像
            cv2.imwrite(f'./result/SNLI/answers/output_image{str(i)}.jpg', img_with_heatmap)
            # print(im.size)
            # plt.imshow(im)  # 设置plt可视化图层为原图
            # attn_map.squeeze()
            # # attn_map = attn_map.resize(im.size)
            # plt.imshow(attn_map.cpu().numpy() * 255, alpha=0.4, cmap='rainbow')  # 这行将attention图叠加显示，透明度0.4
            # plt.axis('off')  # 关闭坐标轴
            # plt.savefig(f'./result/SNLI/answers/output_image{str(i)}.jpg')
            # plt.clf()

            # output_ids = model.generate(
            #     input_ids,
            #     images=image_tensor.unsqueeze(0).half().cuda(),
            #     image_sizes=[img_size],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     # no_repeat_ngram_size=3,
            #     max_new_tokens=1024,
            #     use_cache=False,
            #     output_attentions=True,
            #
            # )

        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print(f"Question: {prompt} Answer: {outputs}")
        # ans_file.write(json.dumps({"img_id": pair_id,
        #                            "prompt": prompt,
        #                            "text": outputs,
        #                            "model_id": model_name,
        #                            "metadata": {}}, ensure_ascii=False) + "\n")
        # ans_file.flush()
    ans_file.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--vision-encoder", type=str, default="clip")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="../../data/SNLI-VE/data/images")
    parser.add_argument("--data-file", type=str, default="../../data/SNLI-VE/data/snli_ve_dev.jsonl")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--projector", type=str, default="./checkpoints/llava-v1.5-13b-pretrain/checkpoint-1000")
    args = parser.parse_args()


    eval_model(args)
