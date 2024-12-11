import os
import torch
import pandas as pd
from transformers import MBartForConditionalGeneration, MBartTokenizer
from sacrebleu import corpus_bleu

# Initialize directories
output_dir = 'mbartTrans'
latest_checkpoint = max(
    [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
    key=os.path.getmtime,
    default=None
)

# Load data
train_df = pd.read_csv("training_data.csv")
parallel_sentences = list(zip(train_df['zh'], train_df['en']))[0:100]
src_sentences, tgt_sentences = zip(*parallel_sentences)

# Load tokens to be added
tokens_to_be_added = []
with open('tokens_to_be_added.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tokens_to_be_added.append(line.strip())

if latest_checkpoint is None:
    # Load model and tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBartTokenizer.from_pretrained(model_name)

    # Add custom tokens and resize model embeddings
    tokenizer.add_tokens(tokens_to_be_added)
    model.resize_token_embeddings(len(tokenizer))
else:
    model = MBartForConditionalGeneration.from_pretrained(latest_checkpoint)
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.add_tokens(tokens_to_be_added)
    model.resize_token_embeddings(len(tokenizer))

# Set source and target languages
tokenizer.src_lang = "zh_CN"
tokenizer.tgt_lang = "en_XX"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters
learning_rate = 5e-5
num_epochs = 3
batch_size = 5
max_length = 256


# Define BLEU score calculation
def calculate_bleu_score(predictions, references):
    """Calculate BLEU score for predictions and references."""
    return corpus_bleu(predictions, [references]).score


# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# RL Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_rewards = []
    batch_losses = []

    for i in range(0, len(src_sentences), batch_size):
        batch_src = src_sentences[i:i + batch_size]
        batch_tgt = tgt_sentences[i:i + batch_size]

        # Tokenize source and target sentences
        src_encodings = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True,
                                  max_length=max_length).to(device)
        tgt_encodings = tokenizer(batch_tgt, return_tensors="pt", padding=True, truncation=True,
                                  max_length=max_length).to(device)

        # Generate translations
        with torch.no_grad():
            outputs = model.generate(**src_encodings, max_length=max_length, num_beams=4)
            predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Calculate BLEU score as the reward
        bleu_score = calculate_bleu_score(predictions, batch_tgt)
        normalized_reward = bleu_score / 100.0  # Normalize the reward
        epoch_rewards.append(normalized_reward)

        # Compute log probabilities of the generated sequence
        outputs = model(**src_encodings, labels=tgt_encodings["input_ids"])
        log_probs = outputs.logits.log_softmax(-1)

        # Get the log probabilities of the target tokens
        target_tokens = tgt_encodings["input_ids"]
        log_probs_of_target = log_probs.gather(2, target_tokens.unsqueeze(-1))

        # Calculate the RL loss (-reward * log probability)
        loss = -torch.mean(normalized_reward * log_probs_of_target.squeeze(-1))

        batch_losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Log batch-level rewards and losses immediately
        print(f"Batch {i // batch_size + 1}: Reward (BLEU) = {bleu_score:.2f}, Loss = {loss.item():.4f}")

    avg_reward = sum(epoch_rewards) / len(epoch_rewards)
    avg_loss = sum(batch_losses) / len(batch_losses)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Average Reward (BLEU): {avg_reward:.2f}")
    print(f"  Average Loss: {avg_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained("mbart-rl-finetuned")
tokenizer.save_pretrained("mbart-rl-finetuned")

