import pdb
import sys
sys.path.append("..")
sys.path.append("../..")
import fitz
import os
import argparse
import uuid
#from utils.utils import new_ip
import glob
#from paddleocr import PaddleOCR, draw_ocr
import json
from utils.gemini import Gemini_Model, prompt_gen_ch_qa_extractive, prompt_gen_ch_qa_vis, prompt_gen_qa, prompt_gen_ch_qa_abstractive, prompt_gen_ch_qa_yes_no
API_KEY = "AIzaSyCYo6MWJKX4nrV8i36GKVVEVeuYfD3co-s"
from dataset.dataloader import DataLoader

def gemini_gen_qa_vision(metadata, prompt, savedir):
    model = Gemini_Model(key=API_KEY)
    data_path = "../../data/pageqa/dev.json"
    img_path = "../../data/pageqa/png"
    savedir = "../../result/gemini/qasper"
    savename = os.path.join(savedir, "en-en.jsonl")
    if os.path.exists(savename):
        with open(savename, "r") as fr:
            line_num = len(fr.readlines())
    else:
        line_num = 0
    dataloader = DataLoader(data_path)
    with open(savename, "w") as f:
        for question, imgname, question_id, answers in dataloader:
            image_path = os.path.join(img_path, imgname + ".png")
            response = model.get_response_vision(image_path, question + "Answer in one sentence:")
            f.write(json.dumps({
                "question_id": question_id,
                "predicted_answer": response,
            }) + "\n")
            print(f"Question: {question} Answer: {response}")

gemini_gen_qa_vision(0, 0, 0)