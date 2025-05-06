import argparse
import pdb

import jieba
import json
from collections import Counter
import string, re

def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def single_f1_zh(gold, answer):
    if type(answer) is list:
        answer = answer[0]
    gold_seg = jieba.cut(gold, cut_all=False)
    answer_seg = jieba.cut(answer, cut_all=False)
    # print("Default Mode: " + "/ ".join(gold_seg))  # 精确模式
    # print("Default Mode: " + "/ ".join(answer_seg))  # 精确模
    gold_seg = "/ ".join(gold_seg).split("/ ")
    answer_seg = "/ ".join(answer_seg).split("/ ")
    common = Counter(answer_seg) & Counter(gold_seg)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(answer_seg)
    recall = num_same / len(gold_seg)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def single_f1_en(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def score_yes_no(prediction):
    total_gold_true, total_predict_true, total_right = 0, 0, 0

    for p in prediction:
        gt_answer = p['gt_answers'] if "gt_answers" in p.keys() else p['gt']
        if type(p['answer']) is list:
            p['answer'] = p['answer'][0]
        p['answer'] = p['answer'].lower()
        if gt_answer.startswith("是") or gt_answer.startswith("是"):
            total_gold_true += 1
        if p['answer'].startswith("是") or p['answer'].startswith("yes"):
            total_predict_true += 1
        if (p['answer'].startswith("是") or p['answer'].startswith("yes")) and (gt_answer.startswith("是") or gt_answer.startswith("yes")):
            total_right += 1
    precision = 1.0 * total_right / total_predict_true
    recall = 1.0 * total_right / total_gold_true
    print(precision, recall)
    print("amount", total_right, total_predict_true, total_gold_true)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="""JSON lines file with each line in format:
                    {'question_id': str, 'predicted_answer': str, 'predicted_evidence': List[str]}"""
    )
    parser.add_argument(
        "--language",
        type=str,
        default="yesno"
    )
    parser.add_argument(
        "--text_evidence_only",
        action="store_true",
        help="If set, the evaluator will ignore evidence in figures and tables while reporting evidence f1"
    )
    args = parser.parse_args()
    extract_scores, abstract_scores, yesno_scores = [], [], []
    if "jsonl" in args.predictions:
        with open(args.predictions, "r") as f:
            predictions = [json.loads(line) for line in f]
    else:
        predictions = json.load(open(args.predictions))
    if args.language == "yesno":
        f1 = score_yes_no(predictions)
        print("Avg F1", f1)
    elif args.language == "mix":
        golds, predicts = [], []
        for p in predictions:
            gold = p['gt_answers'] if "gt_answers" in p.keys() else p['gt']
            predict = p['answer']
            score = eval(f'single_f1_zh')(gold, predict)
            if p['type'] == "extractive":
                extract_scores.append(score)
            elif p['type'] == "abstractive":
                abstract_scores.append(score)
            elif p['type'] == "yesno":
                yesno_scores.append(p)

        print("extract F1", sum(extract_scores) / len(extract_scores))
        print("abs F1", sum(abstract_scores) / len(abstract_scores))
        print("yesno F1", score_yes_no(yesno_scores))
    else:
        scores = []
        for p in predictions:
            gold = p['gt_answers'] if "gt_answers" in p.keys() else p['gt']
            predict = p['answer']
            score = eval(f'single_f1_{args.language}')(gold, predict)
            scores.append(score)

        print("Avg F1", sum(scores) / len(scores))