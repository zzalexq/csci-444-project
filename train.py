import os

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



class WeightedLoss(nn.Module):
    def __init__(self, tokenizer, weight=2.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.weight = weight
        self.pad_token_id = tokenizer.pad_token_id

        # Convert anchor tokens to IDs for source and target
        self.anchor_token_ids = {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in tokens_to_be_added
        }

    def forward(self, labels, logits, input_ids):
        pred = logits.argmax(dim=-1)            # Convert logits to predicted token IDs

        # Cross entropy loss (default)
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.pad_token_id)
        base_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Initialize anchor mask for source and target
        anchor_mask = torch.zeros_like(base_loss)

        # Iterate through anchor token mappings
        for source_token, source_id in self.anchor_token_ids.items():
            target_token = self._map_to_target(source_token)
            target_id = self.tokenizer.convert_tokens_to_ids(target_token)

            # Find where the anchor token appears in source and target
            source_occurrence = (input_ids == source_id).float()
            target_occurrence = (labels == target_id).float()

            # Penalize when source anchor is present but target anchor is not correctly predicted
            mismatched_anchors = source_occurrence * (1 - target_occurrence)
            anchor_mask += mismatched_anchors.view(-1)

        # Apply higher weight to mismatched anchors
        weighted_loss = base_loss + self.weight * anchor_mask
        return weighted_loss.mean()

    def _map_to_target(self, source_token):
        """
        Map source anchor token to its corresponding target anchor token.
        For example: <投资级> -> <investment_grade>.
        """
        # Define your source-to-target mapping here (or pass it as an argument)
        source_to_target_mapping = anchor_words_dict
        return source_to_target_mapping.get(source_token, source_token)  # Default to no change

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load the dataset
dataset = './cleaned_training_data.csv'
df = pd.read_csv(dataset)

# Now split the cleaned dataframe
subset_df = df.sample(n=100000, random_state=42)
train_df, temp_df = train_test_split(subset_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)

# Convert to HuggingFace Dataset objects
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(val_df)


output_dir = 'mbartTrans'
latest_checkpoint = max(
    [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
    key=os.path.getmtime,
    default=None
)

print(f"Latest checkpoint: {latest_checkpoint}")

if latest_checkpoint is None:
    # Load model and tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBartTokenizer.from_pretrained(model_name)

    # Add custom tokens and resize model embeddings
    tokenizer.add_tokens(tokens_to_be_added)
    model.resize_token_embeddings(len(tokenizer))
else:
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(latest_checkpoint)
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(tokens_to_be_added)
    model.resize_token_embeddings(len(tokenizer))


if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
else:
    print("Starting training from scratch.")


# Prepare datasets
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["zh"],
                                max_length=1012,
                                padding="max_length",
                                truncation=True)
    target_encodings = tokenizer(example_batch["en"],
                                 max_length=1012,
                                 padding="max_length",
                                 truncation=True)
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }

train_dataset_tf = train_dataset.map(convert_examples_to_features, batched=True, remove_columns=["zh", "en"])
val_dataset_tf = validation_dataset.map(convert_examples_to_features, batched=True, remove_columns=["zh", "en"])


training_args = Seq2SeqTrainingArguments(
    output_dir='mbartTrans',
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy='epoch',
    save_strategy='steps',
    save_steps=2000,
    logging_steps=100,
    weight_decay=0.01,
    push_to_hub=False,
    learning_rate=2e-5,
    optim="adafactor",
)


seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
weighted_loss = WeightedLoss(tokenizer)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")

        # Compute the weighted loss using our custom WeightedLoss
        loss = weighted_loss(labels, logits, input_ids)  # Pass the input_ids

        return (loss, outputs) if return_outputs else loss


trainer = CustomSeq2SeqTrainer(
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


if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("Starting training from scratch.")
    trainer.train()

# Save the model
trainer.save_model("zhenTranslationMbart")