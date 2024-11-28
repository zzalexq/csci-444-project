from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MBartForConditionalGeneration, MBartTokenizer
from datasets import Dataset
import pandas as pd
import torch

# Load data
tokens_to_be_added = []
with open('tokens_to_be_added.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tokens_to_be_added.append(line.strip())

dataset = './training_data.csv'
dataframe = pd.read_csv(dataset)
train_df = dataframe[:100000]
val_df = dataframe[100000:120000]

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(val_df)

# Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBartTokenizer.from_pretrained(model_name)

# Add custom tokens and resize model embeddings
tokenizer.add_tokens(tokens_to_be_added)
model.resize_token_embeddings(len(tokenizer))

# Prepare datasets
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["zh"], max_length=1024, padding="max_length", truncation=True)
    target_encodings = tokenizer(example_batch["en"], max_length=1024, padding="max_length", truncation=True)
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }

train_dataset_tf = train_dataset.map(convert_examples_to_features, batched=True, remove_columns=["zh", "en"])
val_dataset_tf = validation_dataset.map(convert_examples_to_features, batched=True, remove_columns=["zh", "en"])

# Setup Seq2SeqTrainer
training_args = Seq2SeqTrainingArguments(
    output_dir='mbartTrans',
    num_train_epochs=2,
    per_device_train_batch_size=1,  # Increased batch size
    per_device_eval_batch_size=1,
    evaluation_strategy='steps',
    save_strategy='no',
    eval_steps=2000,
    logging_steps=1000,
    weight_decay=0.01,
    push_to_hub=False,
    learning_rate=2e-5,
    optim="adafactor",
    no_cuda=False  # Ensure CUDA is enabled when available
)

# Data collator for seq2seq tasks
from transformers import DataCollatorForSeq2Seq
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=train_dataset_tf,
    eval_dataset=val_dataset_tf
)

# Check CUDA availability and model device
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Model is on device: {next(model.parameters()).device}")

# Train the model
trainer.train()

# Save the model
trainer.save_model("zhenTranslationMbart")
