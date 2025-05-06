import os
import re
import threading
import time
from typing import List, Tuple
from PIL import Image

import google.generativeai as genai
import tqdm

from utils import load_from_jsonl, append_to_jsonl

DATA_DIR = './'
API_KEY = 'AIzaSyB2rGDZzkVKxgkV8y_uJf4LvK9E9WKfWoE'


def generate_docvqa(model: genai.GenerativeModel, image: Image.Image, QAs: List[Tuple[str, str]]):
    prompt = \
"""You are a document processing specialist. Based on the document scan image and question-answer pairs, help me finish the following jobs:

1. Write a brief description about the topic and key points of the document.
2. From the example question, get the thinking process of the example answer.
3. Imitate the example question-thought-answer tuples and generate 10 more question-thought-answer tuples. Keep the generated answer concise, i.e., using a single word or phrase.

Your output should follow this format, ${...} means the filling content:

Caption: ${brief_description}
"""

    for idx, (question, answer) in enumerate(QAs):
        prompt += \
f"""
Example {idx + 1}:
Q: {question}
T: ${{thought}}
A: {answer}
"""

    prompt += \
"""
Generated 1:
Q: ${question}
T: ${thought}
A: ${answer}

Generated ..."""

    response = model.generate_content([image, prompt])

    return response.text


def generate_chartqa(model: genai.GenerativeModel, image: Image.Image, QAs: List[Tuple[str, str]]):
    prompt = \
"""You are a data analyst. Based on the chart image and question-answer pairs, help me finish the following jobs:

1. Write a brief description about the topic and key points of the chart.
2. Restore the data table corresponding to the chart.
3. From the example question, get the thinking process of the example answer.
4. Imitate the example question-thought-answer tuples and generate 5 more question-thought-answer tuples. Keep the generated answer concise, i.e., using a single word or phrase.

Your output should follow this format, ${...} means the filling content:

Caption: ${brief_description}

Data: ${data_table}
"""

    for idx, (question, answer) in enumerate(QAs):
        prompt += \
f"""
Example {idx + 1}:
Q: {question}
T: ${{thought}}
A: {answer}
"""

    prompt += \
"""
Generated 1:
Q: ${question}
T: ${thought}
A: ${answer}

Generated ..."""

    response = model.generate_content([image, prompt])

    return response.text


def generate_infovqa(model: genai.GenerativeModel, image: Image.Image, QAs: List[Tuple[str, str]]):
    w, h = image.size
    if w > 3072:
        w = 3072
        h = int(h / w * 3072)
        image = image.resize((3072, h), resample=Image.Resampling.BICUBIC)
    crops = []
    crop_hint = ''
    if h > 3072:
        for y in range(0, h, 3072):
            if h - y >= 3072:
                crop = image.crop((0, y, w, y + 3072))
            else:
                crop = Image.new('RGB', (w, 3072))
                crop.paste(image.crop((0, y, w, h)))
            crops.append(crop)
        crop_hint = f' (the whole image is vertically splitted into {len(crops)} sub-images)'

    prompt = \
f"""You are a professional researcher. Based on the infographic image{crop_hint} and question-answer pairs, help me finish the following jobs:

1. Write a brief description about the topic and key points of the infographic.
2. From the example question, get the thinking process of the example answer.
3. Imitate the example question-thought-answer tuples and generate 10 more question-thought-answer tuples. Keep the generated answer concise, i.e., using a single word or phrase.

Your output should follow this format, ${{...}} means the filling content:

Caption: ${{brief_description}}
"""

    for idx, (question, answer) in enumerate(QAs):
        prompt += \
f"""
Example {idx + 1}:
Q: {question}
T: ${{thought}}
A: {answer}
"""

    prompt += \
"""
Generated 1:
Q: ${question}
T: ${thought}
A: ${answer}

Generated ..."""

    if crops:
        response = model.generate_content([*crops, prompt])
    else:
        response = model.generate_content([image, prompt])

    return response.text


def get_structured_extension(text):
    structured_data = {}

    caption_match = re.search(r'Caption:(.+?)\n\n', text, re.DOTALL)
    structured_data['caption'] = caption_match.group(1).strip()

    data_match = re.search(r'Data:\n\n(.+?)\n\n', text, re.DOTALL)
    if data_match:
        structured_data['data'] = data_match.group(1).strip()

    example_matches = re.findall(r'Example \d:\nQ:(.+?)\nT:(.+?)\nA:(.+?)\n\n', text, re.DOTALL)
    structured_data['example'] = []
    for m in example_matches:
        structured_data['example'].append({
            'question': m[0].strip(),
            'thought': m[1].strip(),
            'answer': m[2].strip(),
        })
    assert len(structured_data['example']) > 0

    generated_matches = re.findall(r'Generated \d:\nQ:(.+?)\nT:(.+?)\nA:(.+?)(?=\n\n|\Z)', text, re.DOTALL)
    structured_data['generated'] = []
    for m in generated_matches:
        structured_data['generated'].append({
            'question': m[0].strip(),
            'thought': m[1].strip(),
            'answer': m[2].strip(),
        })
    assert len(structured_data['generated']) > 0

    return structured_data


def generate_conversation(model: genai.GenerativeModel, image: Image.Image):
    w, h = image.size
    if w > 3072:
        w = 3072
        h = int(h / w * 3072)
        image = image.resize((3072, h), resample=Image.Resampling.BICUBIC)
    crops = []
    crop_hint = ''
    if h > 3072:
        for y in range(0, h, 3072):
            if h - y >= 3072:
                crop = image.crop((0, y, w, y + 3072))
            else:
                crop = Image.new('RGB', (w, 3072))
                crop.paste(image.crop((0, y, w, h)))
            crops.append(crop)
        crop_hint = f' (the whole image is vertically splitted into {len(crops)} sub-images)'

    prompt = \
f"""You are a senior researcher, imagine you are talking with a junior researcher about the content in the image{crop_hint}. Think loudly and show me the conversation.

For example, the junior researcher may ask diverse questions and you will give corresponding answers. Questions that have definite answers are preferred. The junior researcher may also ask complex questions, for example, asking you share knowledge about the topic or background stories related to the content in the image. Again, do not ask about uncertain details. Provide detailed answers when answering complex questions, for example, give detailed examples or reasoning steps to make the content more convincing and well-organized. You can include multiple paragraphs if necessary.

Your output should follow this format, ${{...}} means the filling content:

Junior: ${{...}}
Senior: ${{...}}
..."""

    if crops:
        response = model.generate_content([*crops, prompt])
    else:
        response = model.generate_content([image, prompt])

    return response.text


def get_structured_conversation(text):
    structured_data = []

    turns = text.strip().split('Junior:')[1:]
    for i, turn in enumerate(turns):
        turn = turn.strip().split('Senior:')
        if len(turn) == 2:
            structured_data.append({
                'user': turn[0].strip(),
                'asst': turn[1].strip(),
            })
        else:
            assert i == len(turns) - 1, '{}'.format(turn)
    assert len(structured_data) > 0, 'Got empty conversation'

    return structured_data


def generate_and_save(model, dataset, mode, annotation):
    if mode == 'extension':
        json_path = '{}.jsonl'.format(dataset)
    else:
        json_path = '{}_conv.jsonl'.format(dataset)
    image_path = annotation['image']
    QAs = annotation['QAs']
    try:
        image = Image.open(os.path.join(DATA_DIR, image_path))
        if mode == 'extension':
            if dataset == 'docvqa':
                text = generate_docvqa(model, image, QAs)
            elif dataset == 'chartqa':
                text = generate_chartqa(model, image, QAs)
            elif dataset == 'infovqa':
                text = generate_infovqa(model, image, QAs)
            structured = get_structured_extension(text)
        else:
            text = generate_conversation(model, image)
            structured = get_structured_conversation(text)
        append_to_jsonl({'image': image_path, 'gemini': structured}, json_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'WARNING: {e.__class__.__name__} - {image_path}')


if __name__ == '__main__':
    genai.configure(api_key=API_KEY, transport='rest')
    model = genai.GenerativeModel('gemini-pro-vision')

    for dataset in ['docvqa', 'chartqa', 'infovqa']:
        for mode in ['extension', 'conversation']:
            src_json = os.path.join(DATA_DIR, 'ureader_json/train_{}.jsonl'.format(dataset))
            if mode == 'extension':
                dst_json = os.path.join(DATA_DIR, '{}.jsonl'.format(dataset))
            else:
                dst_json = os.path.join(DATA_DIR, '{}_conv.jsonl'.format(dataset))
            data = load_from_jsonl(src_json)
            if os.path.exists(dst_json):
                done = set(_['image'] for _ in load_from_jsonl(dst_json))
            else:
                done = []

            threads = []
            last_time = time.time()
            pbar = tqdm.tqdm(data)
            for i, annotation in enumerate(pbar):
                if annotation['image'] in done:
                    continue
                thread = threading.Thread(target=generate_and_save, args=(model, dataset, mode, annotation))
                time.sleep(max(1.0 - (time.time() - last_time), 0))
                thread.start()
                last_time = time.time()
                threads.append(thread)
                pbar.set_description(f'Active threads: {threading.active_count() - 2}')
            while threading.active_count() - 2 > 0:
                print(f'Active threads: {threading.active_count() - 2}')
                time.sleep(1)
            for thread in threads:
                thread.join()