import json, os

class MathLoader:
    def __init__(self, data_file, img_path):
        with open(data_file, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.img_path = img_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        imgname = item['image']
        imgname = os.path.join(self.img_path, imgname)

        id = item['id']
        return question, imgname, id