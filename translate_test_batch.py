import torch
from datasets import concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

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

from transformers import MBartForConditionalGeneration, MBartTokenizer

# Load mBART model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)  # Move model to MPS
tokenizer = MBartTokenizer.from_pretrained(model_name)

tokenizer.src_lang = "zh_CN"  # Source language (Chinese)
tokenizer.tgt_lang = "en_XX"  # Target language (English)

output_file = "unmodified_model_test_translations.txt"

# Open the file in write mode
with open(output_file, "w", encoding="utf-8") as f:
    print("Starting test_dataset unmodified model translations total examples =", len(test_dataset))
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
        f.write(translated_text + "\n")

        if idx % 100 == 0:
            print("Done with", idx, "out of", len(test_dataset))