import os

import torch
from datasets import concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MBartForConditionalGeneration, MBartTokenizer

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the dataset
dataset = './training_data.csv'
dataframe = pd.read_csv(dataset)

subset_df = dataframe.sample(n=100000, random_state=42)
train_df, temp_df = train_test_split(subset_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)


output_dir = 'mbartTrans'
latest_checkpoint = max(
    [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
    key=os.path.getmtime
)
print(f"Latest checkpoint: {latest_checkpoint}")


# Load model and tokenizer from the latest checkpoint
model = MBartForConditionalGeneration.from_pretrained(latest_checkpoint)
tokenizer = MBartTokenizer.from_pretrained(latest_checkpoint)

tokenizer.src_lang = "zh_CN"  # Source language (Chinese)
tokenizer.tgt_lang = "en_XX"  # Target language (English)

output_file = "modified_model_test_translations.txt"

# Open the file in write mode
with open(output_file, "w", encoding="utf-8") as f:
    print("Starting test_dataset translations total examples =", len(test_dataset))
    for idx, example in enumerate(test_dataset):
        test_sentence = example['zh']
        inputs = tokenizer(
            test_sentence,
            return_tensors="pt",
            max_length=512,
            padding="longest",
            truncation=True
        ).to(device)  # Move inputs to MPS

        # Generate translation
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
            )

        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Write translation to file
        f.write(test_sentence + " ; " + translated_text + "\n")

        if idx % 100 == 0:
            print("Done with", idx, "out of", len(test_dataset))