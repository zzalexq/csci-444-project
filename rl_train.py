import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MBartForConditionalGeneration, MBartTokenizer, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn

# Load data
tokens_to_be_added = []
with open('tokens_to_be_added.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tokens_to_be_added.append(line.strip())

anchor_words_dict = {}
with open('final_anchors.txt', 'r', encoding='utf-8') as f:
    for line in f:
        items = line.split()
        anchor_words_dict[items[1].strip()] = items[0].strip()



# Load the dataset
dataset = './cleaned_training_data.csv'
df = pd.read_csv(dataset)

# Now split the cleaned dataframe
subset_df = df.sample(n=100000, random_state=42)
train_df, temp_df = train_test_split(subset_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)
