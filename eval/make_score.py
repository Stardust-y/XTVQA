
import os
import argparse
import json
import pdb
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from tqdm import tqdm
from typing import Optional



def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        question = prompt

    return question.lower()

def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

def eval_single_chart(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))
    annotations = {(annotation['imgname'], annotation['query'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    acc = []
    for result in results:
        annotation = annotations[(result['img_id'], prompt_processor(result['prompt']))]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": annotation['label'],
        })
        acc.append(1 if result['text'] == annotation['label'] else 0)

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * (sum(acc) / len(acc))))

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a series of point clouds as an animation.")
    parser.add_argument("--mode", type=str, default="ocr")
    parser.add_argument("--annotation_file", type=str, default="./data/ocrvqa/test.jsonl")
    parser.add_argument("--result-file", type=str, default="../data/ch_paper")
    args = parser.parse_args()

    if args.mode == "ocr":
        scores = []
        if "jsonl" in args.result_file:

            with open(args.result_file, "r") as f:

                lines = f.readlines()
                for line in tqdm(lines):
                    line = json.loads(line)
                    scores.append(1.0 if line['gt'] in line['answer'] else 0.0)

        else:
            results = json.load(open(args.result_file, "r"))
            for result in results:
                scores.append(1.0 if result['gt'] in result['answer'] else 0.0)
        accuracy = sum(scores) / len(scores)
        print("Accuracy", accuracy)
    elif args.mode == "chart":
        scores = []
        if "jsonl" in args.result_file:
            with open(args.result_file, "r") as f:

                lines = f.readlines()
                for line in tqdm(lines):
                    line = json.loads(line)
                    if type(line['text']) is list:
                        line['text'] = line['text'][0]

                    score = relaxed_correctness(line['text'].strip(), line['gt'])
                    score = True if line['gt'].strip() in line['text'] else False
                    # print(line['text'], line['gt'], score)
                    # pdb.set_trace()
                    scores.append(score)


        else:
            results = json.load(open(args.result_file, "r"))
            for result in results:
                score = relaxed_correctness(result['text'], result['gt'])
                scores.append(score)
        accuracy = sum(scores) / len(scores)
        print("Accuracy", accuracy)

    else:
        if args.result_file is not None:
            eval(f"eval_single_{args.mode}")(args.annotation_file, args.result_file)

        if args.result_dir is not None:
            for result_file in sorted(os.listdir(args.result_dir)):
                if not result_file.endswith('.jsonl'):
                    print(f'Skipping {result_file}')
                    continue
                eval(f"eval_single_{args.mode}")(args.annotation_file, os.path.join(args.result_dir, result_file))

