import sys, os, json
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

import transformers, torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-v1.6-34b") #echo840/Monkey-Chat  echo840/Monkey
    parser.add_argument("--data_path", type=str, default="./data/pageqa")
    parser.add_argument("--save_path", type=str, default="./result/qasper/ocr/llava-v1.6-34b-2.jsonl")
    parser.add_argument("--metadata", type=str, default="./data/ch_paper/metadata.jsonl")
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--mode", type=str, default="chpaper")

    args = parser.parse_args()
    from dataset.dataloader import PaperTextLoader, LunwenTextLoader
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))
    if args.mode == "enpaper":
        if args.lang == "en":
            dataloader = PaperTextLoader(
                os.path.join(args.data_path, "dev.json"),
                os.path.join(args.data_path, "dev_metadata.json")
            )
        elif args.lang == "zh":
            dataloader = PaperTextLoader(
                os.path.join(args.data_path, "dev_zh_en.jsonl"),
                os.path.join(args.data_path, "dev_metadata.json")
            )
    elif args.mode == "chpaper":
        dataloader = LunwenTextLoader(args.data_path, args.metadata)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    with open(args.save_path, "w") as f:
        for question, imgname, question_id, answers, content in tqdm(dataloader):
            gold = "".join([det[1][0] for det in content[0]])
            if args.mode == "enpaper":
                query = f"Answer the question in a short sentence based on the content. Content:{content}" \
                        f"Question: {question}" \
                        f"Answer in a short sentence: "
            elif args.mode == "chpaper":
                query = f"根据所给内容用一句简短的话回答问题。内容：{content}" \
                        f"问题： {question}" \
                        f"用一句简短的话回答："
            inputs = tokenizer(query, return_tensors="pt")

            # Generate
            generate_ids = model.generate(inputs.input_ids, max_length=4096)
            output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # input_ids = tokenizer.encode(query, return_tensors='pt')
            #
            #
            # # 生成文本回复
            # output = model.generate(input_ids, max_length=4096, num_return_sequences=1)
            #
            # # 解码生成的输出，得到最终的回复文本
            # output_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # 打印回复文本
            print(f"Question {query} Answer: {output_text}")


