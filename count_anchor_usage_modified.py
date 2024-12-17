import re
from collections import defaultdict

# Dictionary mapping anchor words to their translations
anchor_translation_dict = defaultdict(list)
with open('final_anchors.txt', 'r', encoding='utf-8') as f:
    for line in f:
        items = line.split()
        anchor_word = items[1].strip()
        translation = items[0].strip()
        anchor_translation_dict[anchor_word].append(translation)

# Initialize counters
total_anchor_words = 0
correct_translations = 0

# File path
file_path = "final_rl_translations.txt"

# Process the file
with open(file_path, "r", encoding="utf-8") as f:
    # for i, line in zip(range(5), f):
    for line in f:
        # Split the line into original sentence and translated text
        test_sentence, translated_text = line.strip().split(" ; ", 1)

        # Regular expression to find anchor words
        anchor_words = re.findall(r"<([^>]+)>", test_sentence)

        # print(anchor_words)

        # Regular expression to find anchor translations
        anchor_translations = re.findall(r"<([^>]+)>", translated_text)
        # print(anchor_translations)

        # Increment the total count of anchor words
        total_anchor_words += len(anchor_words)

        # Check translations for each anchor word
        for anchor_word in anchor_words:
            # Check if the anchor word is correctly translated
            for anchor_translation in anchor_translation_dict[anchor_word]:
                if anchor_translation in anchor_translations:
                    correct_translations += 1
                    break

# Calculate the fraction of correctly translated anchor words
if total_anchor_words > 0:
    fraction_correct = correct_translations / total_anchor_words
else:
    fraction_correct = 0

# Print results
print(f"Total anchor words: {total_anchor_words}")
print(f"Correct translations: {correct_translations}")
print(f"Fraction of correctly translated anchor words: {fraction_correct:.4f}")
