from dataset import load_dataset

dataset = load_dataset("nyu-mll/glue", "ax")

sub_cat = ["ax", "cola", "mnli", "mnli_matched", "mnli_mismatched", "mrpc", "qnli", "qqp",
"rte", "sst2", "stsb", "wnli"]