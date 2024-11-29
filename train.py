from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MBartForConditionalGeneration, MBartTokenizer
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn

# Load data
tokens_to_be_added = []
with open('tokens_to_be_added.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tokens_to_be_added.append(line.strip())


# Define a custom loss function that focuses on anchor tokens
class WeightedLoss(nn.Module):
    def __init__(self, tokenizer, weight=2.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.weight = weight
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_token_id = tokenizer.convert_tokens_to_ids(
            "<UNK>")  # Replace with your UNK token or any non-relevant token

    def forward(self, labels, logits):
        # Convert logits to token ids
        pred = logits.argmax(dim=-1)

        # Create a mask for anchor tokens (for simplicity, let's assume they're <investment_grade>, <core_business>, etc.)
        anchor_tokens = tokens_to_be_added  # Example anchor tokens
        anchor_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in anchor_tokens]

        # Ensure anchor_ids tensor is on the same device as pred (e.g., MPS, CUDA, or CPU)
        anchor_ids_tensor = torch.tensor(anchor_ids, device=pred.device)

        # Generate mask where anchor tokens are present
        anchor_mask = torch.isin(pred, anchor_ids_tensor).float()

        # Compute the standard cross entropy loss
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Apply higher weights to anchor tokens
        weighted_loss = loss * (1 + self.weight * anchor_mask.view(-1))

        return weighted_loss.mean()


dataset = './training_data.csv'
dataframe = pd.read_csv(dataset)
train_df = dataframe[:80000]
val_df = dataframe[80000:10000]

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
    input_encodings = tokenizer(example_batch["zh"], padding="longest", truncation=True)
    target_encodings = tokenizer(example_batch["en"], padding="longest", truncation=True)
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    # Apply your anchor token mask here to evaluate how many anchor tokens are predicted correctly
    anchor_tokens = tokens_to_be_added
    anchor_ids = [tokenizer.convert_tokens_to_ids(t) for t in anchor_tokens]

    correct = 0
    total = 0

    for label, pred in zip(labels, preds):
        for l, p in zip(label, pred):
            if l in anchor_ids:
                total += 1
                if l == p:
                    correct += 1
    accuracy = correct / total if total > 0 else 0
    return {"anchor_accuracy": accuracy}

train_dataset_tf = train_dataset.map(convert_examples_to_features, batched=True, remove_columns=["zh", "en"])
val_dataset_tf = validation_dataset.map(convert_examples_to_features, batched=True, remove_columns=["zh", "en"])

# Setup Seq2SeqTrainer
training_args = Seq2SeqTrainingArguments(
    output_dir='mbartTrans',
    num_train_epochs=2,
    per_device_train_batch_size=1,  # Increased batch size
    per_device_eval_batch_size=1,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_steps=1000,
    weight_decay=0.01,
    push_to_hub=False,
    learning_rate=2e-5,
    optim="adafactor",
)

# Data collator for seq2seq tasks
from transformers import DataCollatorForSeq2Seq
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize the Trainer
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     tokenizer=tokenizer,
#     data_collator=seq2seq_data_collator,
#     train_dataset=train_dataset_tf,
#     eval_dataset=val_dataset_tf,
#     compute_metrics=compute_metrics
# )
#
weighted_loss = WeightedLoss(tokenizer)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        # Compute the weighted loss using our custom WeightedLoss
        loss = weighted_loss(labels, logits)

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

# Train the model
trainer.train()

# Save the model
trainer.save_model("zhenTranslationMbart")