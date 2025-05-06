"""
Official script for evaluating models built for the Qasper dataset. The script
outputs Answer F1 and Evidence F1 reported in the paper.
"""
from collections import Counter
import argparse
import string
import re
import json


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


def token_f1_score(prediction, ground_truth):
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
    # print(precision, recall)
    return f1


def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_answers_and_evidence(data, text_evidence_only):
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(f"Annotation {answer_info['annotation_id']} does not contain an answer")
                    if text_evidence_only:
                        evidence = [text for text in answer_info["evidence"] if "FLOAT SELECTED" not in text]
                    else:
                        evidence = answer_info["evidence"]
                    references.append({"answer": answer, "evidence": evidence, "type": answer_type})
            answers_and_evidence[question_id] = references

    return answers_and_evidence


def evaluate(gold, predicted):
    max_answer_f1s = []
    max_evidence_f1s = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    num_missing_predictions = 0
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            # max_answer_f1s.append(0.0)
            # max_evidence_f1s.append(0.0)
            continue
        answer_f1s_and_types = [
            (token_f1_score(predicted[question_id]["answer"], reference["answer"]),
             reference["type"])
            for reference in gold[question_id]
        ]
        max_answer_f1, answer_type = sorted(answer_f1s_and_types, key=lambda x: x[0], reverse=True)[0]
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)
        evidence_f1s = [
            paragraph_f1_score(predicted[question_id]["evidence"], reference["evidence"])
            for reference in gold[question_id]
        ]
        max_evidence_f1s.append(max(evidence_f1s))

    print(len(max_answer_f1s))
    mean = lambda x: sum(x) / len(x) if x else 0.0
    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {key: mean(value) for key, value in max_answer_f1s_by_type.items()},
        "Evidence F1": mean(max_evidence_f1s),
        "Missing predictions": num_missing_predictions
    }

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
        "--gold",
        type=str,
        required=True,
        help="Test or dev set from the released dataset"
    )
    parser.add_argument(
        "--text_evidence_only",
        action="store_true",
        help="If set, the evaluator will ignore evidence in figures and tables while reporting evidence f1"
    )
    args = parser.parse_args()
    gold_data = json.load(open(args.gold))
    gold_answers_and_evidence = get_answers_and_evidence(gold_data, args.text_evidence_only)
    predicted_answers_and_evidence = {}
    if "jsonl" in args.predictions:
        for line in open(args.predictions):
            prediction_data = json.loads(line)
            d = prediction_data
            if d['answer'].startswith("Yes") or d['answer'].startswith("是"):
                answer = "Yes"
            elif d['answer'].startswith("No") or d['answer'].startswith("否") or d['answer'].startswith("不"):
                answer = "No"
            else:
                answer = prediction_data['answer']
            predicted_answers_and_evidence[prediction_data["question_id"]] = {
                "answer": answer,
                "evidence": [],
                #"evidence": prediction_data["predicted_evidence"]
            }
    else:
        prediction_data = json.load(open(args.predictions, "r"))
        for d in prediction_data:
            if d['answer'].startswith("Yes") or d['answer'].startswith("是"):
                answer = "Yes"
            elif d['answer'].startswith("No") or d['answer'].startswith("否") or d['answer'].startswith("不"):
                answer = "No"
            else:
                answer = d['answer']
            predicted_answers_and_evidence[d['question_id']] = {
                "answer": answer,
                "evidence": [],
            }
    evaluation_output = evaluate(gold_answers_and_evidence, predicted_answers_and_evidence)
    print(json.dumps(evaluation_output, indent=2))