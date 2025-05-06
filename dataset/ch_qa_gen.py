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
# from utils.gemini import Gemini_Model, prompt_gen_ch_qa_extractive, prompt_gen_ch_qa_vis, prompt_gen_qa, prompt_gen_ch_qa_abstractive, prompt_gen_ch_qa_yes_no
# API_KEY = "AIzaSyB2rGDZzkVKxgkV8y_uJf4LvK9E9WKfWoE"


def paddle_pdfs(data_dir, save_path="../data"):
    from paddleocr import PaddleOCR, draw_ocr
    lunwen_list = os.listdir(os.path.join(data_dir, "pdf"))
    ocr = PaddleOCR(lang="ch")
    with open(os.path.join(save_path, "metadata.jsonl"), "w") as f:
        for lid, lunwen in enumerate(lunwen_list):
            doc = fitz.open(os.path.join(os.path.join(data_dir, "pdf"), lunwen))
            img_folder = os.path.join(data_dir, f"png/{lid}")
            print("img folder", img_folder)
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            for i, page in enumerate(doc):
                zoom_x = 2  # (1.33333333-->1056x816)   (2-->1584x1224)
                zoom_y = 2
                mat = fitz.Matrix(zoom_x, zoom_y)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_dir = os.path.join(img_folder, f"{str(i)}.png")
                pix.save(img_dir)
                print("saved to", img_dir)
                result = ocr.ocr(img_dir)
                print("lid", lid)
                f.write(json.dumps({"lid": lid, "origin_name": lunwen, "image_path": img_dir, "detection": result}, ensure_ascii=False) + "\n")

def paddle_png(img_path):
    ocr = PaddleOCR(lang="en")
    result = ocr.ocr(img_path)
    return result


def gemini_gen_qa_text(metadata, prompt, savedir="../data/ch_paper/qa"):
    model = Gemini_Model(key="AIzaSyCYo6MWJKX4nrV8i36GKVVEVeuYfD3co-s", vision=False)
    print("model initialized")
    savename = os.path.join(savedir, "zh-zh-gemini-txt-yes-no.jsonl")
    if os.path.exists(savename):
        with open(savename, "r") as fr:
            line_num = len(fr.readlines())
    else:
        line_num = 0
    index  = 0
    with open(metadata, "r") as f:
        for i, line in enumerate(f):
            if i < line_num:
                continue
            ocr_res = eval(line)
            detection = ocr_res['detection'][0]
            img_path = ocr_res['image_path']

            if len(detection) > 10: ## 检测出的文字大于一定数量 可以提问
                text = "".join([det[1][0] for det in detection])
                query = prompt + text
                try:
                    response = model.get_response_text(query)
                except:
                    print("skip")
                    pdb.set_trace()
                    continue
                print(response.text, "-" * 20)
                with open(savename, "a") as fr:
                    for line in response.text.split("\n"):
                        try:
                            qa = eval(line.strip())
                            fr.write(json.dumps({
                                "question_id": index,
                                "img_path": img_path,
                                "question": qa['question'],
                                "answer": qa['answer']
                            }, ensure_ascii=False) + "\n")
                            index += 1
                        except:
                            pass

def gemini_gen_qa_vision(metadata, prompt, savedir):
    model = Gemini_Model(key=API_KEY)
    savename = os.path.join(savedir, "zh-zh-gemini-vis.jsonl")
    if os.path.exists(savename):
        with open(savename, "r") as fr:
            line_num = len(fr.readlines())
    else:
        line_num = 0

    with open(metadata, "r") as f:
        for i, line in enumerate(f):
            if i < line_num:
                continue
            ocr_res = eval(line)
           # detection = ocr_res['detection'][0]
            image_path = ocr_res['image_path']
            query = prompt

            response = model.get_response_vision(image_path, query)
            ocr_res['qas'] = []
            for line in response.split("\n"):
                try:
                    qa = eval(line.strip())
                    print(qa)
                    ocr_res['qas'].append(qa)
                except:
                    ocr_res['qas'].append(line)
            with open(os.path.join(savedir, "zh-zh-gemini-vis.jsonl"), "a") as fr:
                fr.write(json.dumps(ocr_res, ensure_ascii=False)+"\n")

def gemini_ocr(args):
    model = Gemini_Model(key=API_KEY)
    savename = os.path.join(args.savedir, "zh-gemini-ocr.jsonl")
    imgs = glob.glob(os.path.join(args.image_path, os.path.join("*", "*.png")))
    if os.path.exists(savename):
        with open(savename, "r") as fr:
            line_num = len(fr.readlines())
    else:
        line_num = 0
    with open(savename, "w") as f:
        for img in imgs:
            response = model.get_response_vision(img, "识别图中所有文字")
            print(response)
            f.write(json.dumps({
                "img_path": img,
                "predicted_answer": response,
            }) + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize a series of point clouds as an animation.")
    parser.add_argument("--vis", type=bool, default=False)
    parser.add_argument("--savedir", type=str, default="../result/ocr/")
    parser.add_argument("--datadir", type=str, default="../data/ch_paper")
    parser.add_argument("--image_path", type=str, default="../data/ch_paper/png")
    parser.add_argument("--form", type=str, default="yes_no")
    parser.add_argument("--mode", type=str, default="enpaper")
    args = parser.parse_args()
    from paddleocr import PaddleOCR, draw_ocr
    res = {}
    if args.mode == "enpaper":
        dev_data = json.load(open("../data/pageqa/dev.json"))
        with open("../data/pageqa/dev_metadata_en.json", "w") as f:
            for data in dev_data:
                imgpath = f"../data/pageqa/png/{data['imgname']}.png"
                result = paddle_png(imgpath)
                res[data['imgname']] = result
                json.dump(res, f, indent=4)


    else:

        gemini_ocr(args)
        exit()
        metadata = os.path.join(args.savedir, "metadata.jsonl")

        if not os.path.exists(metadata):
            paddle_pdfs(args.datadir, args.savedir)

        if args.vis:
            print("using vis---")
            prompt = prompt_gen_ch_qa_vis
            gemini_gen_qa_vision(metadata, prompt, args.savedir)
        else:
            prompt = eval(f'prompt_gen_ch_qa_{args.form}')
            gemini_gen_qa_text(metadata, prompt, args.savedir)

