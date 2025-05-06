import os
import pdb
import time

import google.generativeai as genai
from pathlib import Path

generation_config = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}
prompt_gen_qa = "You are a scientific literature Q&A expert on data set processing. Your task is to generate questions that are appropriate as answers to the data set, based on the literature images I've given you. Keep the questions as academic as possible. There can be only one question in a sentence. The generated questions must be detailed and valuable, and do not generate particularly macro questions." \
                "You must follow the format of the Q&A I gave you: {'question': <your question here>, 'answer': <your answer here>}" \
                "You must use english to ask question. All questions and answers should be performed in english" \
                "Here's my content:"

prompt_gen_ch_qa_extractive = "你是一个科学文献问答对数据集处理专家。你的任务是根据我给出的文献图片，生成适合作为问答对数据集的问题。问题要尽量学术。一句话中只能有一个问题。生成的问题必须细节、有价值，不要生成特别宏观的问题。" \
                   "你的问题需要是抽取式的，即答案必须是文中出现的一个数字，词语或者短句。" \
                   "你必须按照我给出的问答对格式来生成：{\"question\": <your question here>, \"answer\": <your answer here>}" \
                   "请对特殊符号进行转义" \
                   "我的内容如下："
prompt_gen_ch_qa_abstractive = "你是一个科学文献问答对数据集处理专家。你的任务是根据我给出的文献图片，生成适合作为问答对数据集的问题。问题要尽量学术。一句话中只能有一个问题。生成的问题要能概括捕捉文本中的要点和主旨,而不只是浅层次的细节。" \
                   "你的问题需要是摘要式的，即答案不是直接引用文章中的原文，而是需要一定推理、归纳、总结才能获得答案。"  \
                   "你必须按照我给出的问答对格式来生成：{\"question\": <your question here>, \"answer\": <your answer here>}" \
                   "请对特殊符号进行转义" \
                   "我的内容如下："

prompt_gen_ch_qa_yes_no = "你是一个科学文献问答对数据集处理专家。你的任务是根据我给出的文献图片，生成适合作为问答对数据集的问题。问题要尽量学术。一句话中只能有一个问题。" \
                   "你的问题需要是判断对错式的，对原文中的事实性知识做出判断或者判断需要推理的问题，答案只能是是或否。"  \
                   "你必须按照我给出的问答对格式来生成：{\"question\": <your question here>, \"answer\": <your answer here>}" \
                   "请对特殊符号进行转义" \
                   "我的内容如下："

prompt_gen_qa_vis = "You are a scientific literature Q&A expert on data set processing. Your task is to generate questions that are appropriate as answers to the data set, based on the literature images I've given you. Keep the questions as academic as possible. There can be only one question in a sentence. The generated questions must be detailed and valuable, and do not generate particularly macro questions." \
                "You must follow the format of the Q&A I gave you: {'question': <your question here>, 'answer': <your answer here>}" \
                "Here's my content:"

prompt_gen_ch_qa_vis = "你是一个科学文献问答对数据集处理专家。你的任务是根据我给出的文献图片，生成适合作为问答对数据集的问题。问题要尽量学术。一句话中只能有一个问题。生成的问题必须细节、有价值，不要生成特别宏观的问题。" \
                   "你必须按照我给出的问答对格式来生成：{\"question\": <your question here>, \"answer\": <your answer here>}" \
                       "如：{\"question\": \"西气东输郑州黄河穿越顶管工程管道顶进工艺有哪些特点？\", \"answer\": \"采用分段顶进、浮法顶进、定向纠偏等工艺，确保管道顶进的安全性、精度和效率。\"}" \
                       "{\"question\": \"DS-3系统中，星间距的变化范围是多少？\", \"answer\": \"50米到1千米\"}" \

safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


# verify response
def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        # print("Response Error")
        return False
    return True


class Gemini_Model():
    def __init__(self, key, vision=True, patience=1, sleep_time=1):
        print("!!Bard Token:", key)
        self.patience = patience
        self.sleep_time = sleep_time
        genai.configure(api_key=key)
        self.vision = vision
        if self.vision:
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-exp-0827",
                              generation_config=generation_config,
                              safety_settings=safe)
        else:
            self.model = genai.GenerativeModel(model_name="gemini-pro",
                                               generation_config=generation_config,
                                               safety_settings=safe)

    def get_response_text(self, prompt):
        response = self.model.generate_content(prompt)
        return response
    def get_response_vision(self, image_path, input_text):
        patience = self.patience
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                # Validate that an image is present
                print("img", image_path)
                if not (img := Path(image_path)).exists():
                    raise FileNotFoundError(f"Could not find image: {img}")

                print(type(Path(image_path).read_bytes()))

                image_parts = [
                    {
                        "mime_type": "image/jpeg",
                        "data": Path(image_path).read_bytes()
                    },
                ]

                prompt_parts = [
                    input_text,
                    image_parts[0],
                ]

                response = self.model.generate_content(prompt_parts)
                print(response)
                # print("feedback", response.prompt_feedback)
                if response.prompt_feedback:
                    return "None"
                return response.text


            except Exception as e:
                print(e)
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""

if __name__ == "__main__":
    model = Gemini_Model(key="AIzaSyCYo6MWJKX4nrV8i36GKVVEVeuYfD3co-s", vision=False)
    response = model.get_response_text("hello")
    print(response)