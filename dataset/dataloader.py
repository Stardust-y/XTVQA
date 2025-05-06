import json

import glob
import os

class MathLoader:
    def __init__(self, img_path, data_file):
        with open(data_file, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.img_path = img_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        options = item['options']
        imgname = item['image']
        imgname = os.path.join(self.img_path, imgname)
        if len(options):
            question += f"options from the following options: {str(options)}"

        id = item['id']
        return {
            "question_id": id,
            "image_path": imgname,
            "question": question,
        }

class PaperLoader:
    def __init__(self, data_file):
        if "jsonl" in data_file:
            with open(data_file, "r") as f:
                self.data = [json.loads(line) for line in f]
        else:
            self.data = json.load(open(data_file, "r"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        imgname = item["imgname"]
        question_id = item["question_id"]
        answers = item["answers"]
        return question, imgname, question_id, answers

class PaperTextLoader:
    def __init__(self, data_file, metadata_file):
        if "jsonl" in data_file:
            with open(data_file, "r") as f:
                self.data = [json.loads(line) for line in f]
        else:
            self.data = json.load(open(data_file, "r"))
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        imgname = item["imgname"]
        question_id = item["question_id"]
        answers = item["answers"]
        content = self.metadata['imgname']
        return question, imgname, question_id, answers, content

class OcrvqaLoader:
    def __init__(self, data_file):
        with open(data_file, "r") as f:
            self.data = [json.loads(line) for line in f]
        # self.test_data = []
        # for k, v in self.data.items():
        #     if v['split'] == 3:
        #         for q, a in zip(v['questions'], v['answers']):
        #             self.test_data.append({
        #                 "imgname": k,
        #                 "question": q,
        #                 "answer": a
        #             })
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question_id = item['question_id']
        question = item['question']
        answer = item['answer']
        imgname = item['imgname']
        return question_id, question, imgname, answer

class TextvqaLoader:
    def __init__(self, data_file):
        if "jsonl" in data_file:
            with open(data_file, "r") as f:
                self.data = [json.loads(line) for line in f]
        else:
            self.data = json.load(open(data_file, "r"))
            self.data = self.data['data']
        with open(os.path.join(os.path.dirname(data_file), "TextVQA_Rosetta_OCR_v0.2_val.json"), "r") as f:
            self.ocr_data = json.load(f)['data']
            self.ocr_data = {data['image_id']: data['ocr_tokens'] for data in self.ocr_data}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        imgname = item["image_id"]
        question_id = item["question_id"]
        answers = item["answers"][0]
        ocr_tokens = self.ocr_data[imgname]
        return question, imgname, question_id, answers, ocr_tokens

class ChartvqaLoader:
    def __init__(self, data_file):
        if "jsonl" in data_file:
            with open(data_file, "r") as f:
                self.data = [json.loads(line) for line in f]
        else:
            self.data = json.load(open(data_file, "r"))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["query"]
        imgname = item["imgname"]
        answers = item["label"]
        return question, imgname, answers

class DocvqaLoader:
    def __init__(self, data_file):
        if "jsonl" in data_file:
            with open(data_file, "r") as f:
                self.data = [json.loads(line) for line in f]
        else:
            self.data = json.load(open(data_file, "r"))
            self.data = self.data['data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question_id = item['questionId']
        question = item["question"]
        image = item["image"]
        answers = item["answers"][0]
        return question_id, question, image, answers

class LunwenLoader:
    def __init__(self, data_file):
        self.data = []
        with open(data_file, "r") as f:
            for line in f:
                self.data.append(eval(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        question = item['question']
        answer = item['answer']

        return {
            "img_path": img_path,
            "question": question,
            "answer": answer
        }
class LunwenTextLoader:
    def __init__(self, data_file, metadata_file):
        self.data, self.metadata = [], {}
        with open(data_file, "r") as f:
            for line in f:
                self.data.append(eval(line))
        with open(metadata_file, "r") as f:
            for line in f:
                line_data = json.loads(line)
                self.metadata[line_data['image_path']] = line_data['detection']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        question = item['question']
        answers = item['answer']
        question_id = item['question_id']
        content = self.metadata[img_path]

        return question, img_path, question_id, answers, content


if __name__ == "__main__":
    from paddleocr import PaddleOCR, draw_ocr
    ocr = PaddleOCR(lang="en")
    dataloader = DataLoader("./data/pageqa/dev_zh_en.jsonl")
    imgs = glob.glob(os.path.join("./data/pageqa/png", "*.png"))
    with open("./data/pageqa/ocr.jsonl", "w") as f:
        for img in imgs:
            result = ocr.ocr(img)
            f.write(json.dumps({
                "imgname": img.split('/')[-1].replace(".png", ""),
                "ocr": result
            }) + "\n")



