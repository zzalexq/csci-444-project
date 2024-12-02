import os
import numpy as np
import sacrebleu
from transformers import Seq2SeqTrainingArguments, MBartForConditionalGeneration, MBartTokenizer, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


# Check CUDA availability and model device
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define BLEU-based reward system and anchor adjustment
class WeightedLossRL(nn.Module):
    def __init__(self, tokenizer, anchor_weights, weight=2.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.weight = weight
        self.pad_token_id = tokenizer.pad_token_id
        self.anchor_weights = anchor_weights

    def forward(self, labels, logits, input_ids):
        pred = logits.argmax(dim=-1)
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.pad_token_id)
        base_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        anchor_mask = torch.zeros_like(base_loss)

        for token, weight in self.anchor_weights.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            occurrence = (input_ids == token_id).float()
            anchor_mask += weight * occurrence.view(-1)

        weighted_loss = base_loss + self.weight * anchor_mask
        return weighted_loss.mean()


# RL-based training loop
class RLTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, anchor_weights, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.anchor_weights = anchor_weights
        self.training_args = training_args
        self.data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        self.loss_fn = WeightedLossRL(tokenizer, anchor_weights)
        self.optimizer = AdamW(model.parameters(), lr=self.training_args.learning_rate)

    def compute_bleu_reward(self, references, hypotheses):
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score

    def train_step(self, batch):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        input_ids = inputs["input_ids"]

        # Generate translations for reward calculation
        generated_ids = self.model.generate(input_ids)
        generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        reference_texts = [self.tokenizer.decode(l, skip_special_tokens=True) for l in inputs["labels"]]

        # Compute BLEU-based rewards
        rewards = [self.compute_bleu_reward([ref], [gen]) for ref, gen in zip(reference_texts, generated_texts)]

        # Adjust anchor weights
        for token, weight in self.anchor_weights.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            token_occurrence = (input_ids == token_id).float()
            weight_update = np.mean(rewards) * token_occurrence.mean().item()
            self.anchor_weights[token] += weight_update

        # Compute weighted loss with adjusted anchor weights
        loss = self.loss_fn(inputs["labels"], logits, input_ids)
        return loss

    def train(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.training_args.per_device_train_batch_size,
        )

        for epoch in range(self.training_args.num_train_epochs):
            print(f"Epoch {epoch + 1}/{self.training_args.num_train_epochs}")
            
            # Wrap the dataloader with tqdm for batch progress tracking
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", leave=False)
            
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update progress bar with the current loss
                progress_bar.set_postfix({"Loss": loss.item()})

                if step % self.training_args.logging_steps == 0:
                    print(f"Step {step}, Loss: {loss.item()}")

            progress_bar.close()


# Load data
dataframe = pd.read_csv('./training_data.csv')
train_df, temp_df = train_test_split(dataframe.sample(n=100000, random_state=42), test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)

# Convert pandas DataFrame to Dataset, resetting index to avoid unexpected columns
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(val_df)

print("Train dataset fields:", train_dataset.features)
print("Validation dataset fields:", validation_dataset.features)


# Add custom tokens and resize embeddings
with open('tokens_to_be_added.txt', 'r', encoding='utf-8') as f:
    tokens_to_be_added = [line.strip() for line in f]

# Initialize anchor weights
anchor_weights = {token: 1.0 for token in tokens_to_be_added}

output_dir = 'mbartTrans-rl'
latest_checkpoint = max(
    [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
    key=os.path.getmtime,
    default=None
)

print(f"Latest checkpoint: {latest_checkpoint}")

if latest_checkpoint is None:
    # Load model and tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = MBartTokenizer.from_pretrained(model_name)

    # Add custom tokens and resize model embeddings
    tokenizer.add_tokens(tokens_to_be_added)
    model.resize_token_embeddings(len(tokenizer))
else:
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(latest_checkpoint).to(device)
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(tokens_to_be_added)
    model.resize_token_embeddings(len(tokenizer))


if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
else:
    print("Starting training from scratch.")


# Convert data to features
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["zh"], max_length=512, padding="max_length", truncation=True)
    target_encodings = tokenizer(example_batch["en"], max_length=512, padding="max_length", truncation=True)
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }

train_dataset = train_dataset.map(convert_examples_to_features, batched=True, remove_columns=["zh", "en"])
val_dataset = validation_dataset.map(convert_examples_to_features, batched=True, remove_columns=["zh", "en"])


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='mbartTrans-rl',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy='epoch',
    save_strategy='steps',
    save_steps=1000,
    logging_steps=1000,
    learning_rate=2e-5,
    optim="adafactor"
)

# Train the model with RL
rl_trainer = RLTrainer(model, tokenizer, train_dataset, val_dataset, anchor_weights, training_args)
if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    rl_trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("Starting training from scratch.")
    rl_trainer.train()

# Save the model
model.save_pretrained("zhenTranslationMbartRL")
tokenizer.save_pretrained("zhenTranslationMbartRL")
