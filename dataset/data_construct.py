

import pdb

import fitz
import json
import os
### READ IN PDF
from collections import Counter
LAST_INDEX = 18
with open("../data/qasper-train-dev-v0.3/qasper-train-v0.3.json", "r") as f:
    data = json.load(f)

save_dir = "../data/pageqa/train.json"
if os.path.exists(save_dir):
    with open(save_dir, "r") as f:
        qa_list = json.load(f)
else:
    qa_list = []
print([qa['question_id'] for qa in qa_list])

import uuid
black_list = ["1802.00396", "1911.03343", "1908.07822", "1612.04675", "1702.03274"]
def hl_section(hl_evidence, full_text):
    hl_evidence = hl_evidence.strip()
    for section in full_text:
        section_name = section['section_name']
        paragraphs = "\n".join(section['paragraphs'])
        if hl_evidence in paragraphs:
            print("sn", section_name)
            section_name = section_name.split(" ::: ")[-1]
            return section_name
    return None

counter = 0
skip_num = 0
for num, d in enumerate(data.items()):
    # if num < 247:
    #     print("computed", num)
    #     continue
    k, v = d
    if k in black_list:
        print(k, "in black list")
        continue
    print("num", num, "doc", k)
    pdf_url = f"https://arxiv.org/pdf/{k}"
    output_path = f"../data/pageqa/pdfs/{k}.pdf"
    if not os.path.exists(output_path):
        print(f"wget {pdf_url} -O {output_path}")
        os.system(f"wget {pdf_url} -O {output_path}")
    doc = fitz.open(output_path)
    qas, full_text = v['qas'], v['full_text']

    for qa in qas:
        question_id = qa['question_id']
        # print(question_id, [qa['question_id'] for qa in qa_list])
        if question_id in [qa['question_id'] for qa in qa_list]:
            print(question_id, "has been calculated, skip")
            continue

        answers = qa['answers']
        sections = []  #majority vote 选section
        answer_dict = {"extractive_spans": [],
                       "yes_no": [],
                       "free_form_answer": []}
        for answer in answers:
            answer = answer['answer']
            if answer['unanswerable']:
                continue
            hl_evidence = answer['highlighted_evidence']
            if len(hl_evidence) == 0:
                print("answer", answer)
                continue
            # for evidence in hl_evidence:
            #     section = hl_section(hl_evidence[0], full_text)
            #     if section is not None:
            #         sections.append(section)
            answer_dict['extractive_spans'].append(answer['extractive_spans'])
            answer_dict['yes_no'].append(answer['yes_no'])
            answer_dict['free_form_answer'].append(answer['free_form_answer'])
        # 创建Counter对象
        # print("sections", sections)
        # counter = Counter(sections)
        # if len(sections) == 0:
        #     continue   ## 没找到对应的section
        # top_section = counter.most_common(1)[0][0]
        candidates = []
        for i, page in enumerate(doc):
            text_instances = []
            for he in hl_evidence:  #在这一页中通过hl第一句话找所有hl是否有出现
                he = he.split(".")[0]
                if he == "":
                    continue
                text_instance = page.search_for(he)
                if text_instance is None:
                    continue
                else:
                    text_instances += page.search_for(he)
               # print(text_instances)
            if len(text_instances):
                candidates.append(page)
        print("top", len(candidates))
        if len(candidates) > 0:
            page = candidates[0]
        else:
            for i, page in enumerate(doc):
                text_instances = []
                # pdb.set_trace()
                for he in hl_evidence:
                    he = he.split(".")[0].split(",")[0]
                    text_instance = page.search_for(he)
                    if text_instance is None:
                        continue
                    else:
                        text_instances += page.search_for(he)
                    # print(text_instances)
                if len(text_instances):
                    candidates.append(page)
        if len(candidates) > 0:
            page = candidates[0]
        else:
            skip_num += 1
            print("unsuccessful so far", skip_num)
            continue
        zoom_x = 2  # (1.33333333-->1056x816)   (2-->1584x1224)
        zoom_y = 2
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        unique_id = str(uuid.uuid4())[:12]  # 获取UUID的前9位作为哈希码
        pix.save("../data/pageqa/png/" + unique_id + ".png")
        print(question_id, "save to", "../data/pageqa/png/" + unique_id + ".png")
        qa_list.append({
            "question": qa['question'],
            "imgname": unique_id,
            "question_id": qa['question_id'],
            "answers": answer_dict,
        })
    with open(save_dir, "w") as f:
        f.write(json.dumps(qa_list, indent=4))


