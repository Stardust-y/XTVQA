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
from torch.utils.data import Dataset, DataLoader
import os
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoModelForCausalLM, LlamaTokenizer

from PIL import Image
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
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        img_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        print(image_tensor.shape)
        exit()
        return input_ids, image_tensor, prompt, img_size, pair_id

    def __len__(self):
        return len(self.data)
def create_data_loader(data_file, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(data_file, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-v1.5-13b") #echo840/Monkey-Chat  echo840/Monkey
    parser.add_argument("--data_path", type=str, default="./data/pageqa")
    parser.add_argument("--save_path", type=str, default="./result/qasper/llava-v1.5-13b.jsonl")
    parser.add_argument("--conv-mode", type=str, default="chatml_direct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()


    if "llava" in args.model_path:
        checkpoint = args.model_path

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name)
        data_loader = create_data_loader(args.data_file, args.image_folder, tokenizer, image_processor, model.config)

        with open(args.save_path, "w") as f:
            for question, imgname, question_id, answers in data_loader:
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
                # print("img tensor", image_tensor)
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
                }) + "\n")
                print(f"Question: {question} Answer: {outputs}")



