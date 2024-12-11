import os

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MBartForConditionalGeneration, MBartTokenizer, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn


model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBartTokenizer.from_pretrained(model_name)
# tokenizer.add_tokens(tokens_to_be_added)
# model.resize_token_embeddings(len(tokenizer))


