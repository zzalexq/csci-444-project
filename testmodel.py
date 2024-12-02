import os

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MBartForConditionalGeneration, MBartTokenizer

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

# Test a sample input sentence
test_sentence = "总部位于香港的来宝集团近来忙于筹集资金，以免失去其<投资级>信用评级，这种评级对于其<核心业务>（在世界各地从事大量<原材料>的贸易）的盈利能力至关重要。"  # Replace with your test Chinese sentence
inputs = tokenizer(test_sentence, return_tensors="pt", padding="longest", truncation=True)

# Generate translation
model.eval()
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1024,
        num_beams=4  # Optional: use beam search for better quality
    )


# Decode and print the result
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Translated text: {translated_text}")