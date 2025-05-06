import json
import os, pdb
import random

def get_index_by_category(category="cs"):
    index_list = []
    with open("./data/arxiv-metadata-oai-snapshot.json", "r") as f:
        line = f.readline()
        while line:
            data = json.loads(line)
            print(data['categories'])
            if data['categories'][:2] == "cs":
                index_list.append(data['id'])
            line = f.readline()
    return index_list


if __name__ == "__main__":
    category = "cs"
    # index_list = get_index_by_category("cs")
    with open(f"../data/paperocr/cs.txt", "r") as f:
        index_list = f.readlines()
    random.shuffle(index_list)

    # 从打乱后的列表中选取前1000个元素
    index_list = index_list[:1000]
    for id in index_list:
        cmd = f"wget https://ar5iv.labs.arxiv.org/html/{id.strip()}#/ -O ../data/paperocr/html/{id.strip()}.html"
        cmd_pdf = f"wget https://arxiv.org/pdf/{id.strip()}#/ -O ../data/paperocr/html/{id.strip()}.pdf"
        print(cmd)
        try:
            os.system(cmd)
            os.system(cmd_pdf)
        except:
            continue