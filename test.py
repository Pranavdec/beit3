from datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="data",
    tokenizer=tokenizer,
    annotation_data_path="data/vqa",
)