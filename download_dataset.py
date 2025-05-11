from datasets import load_dataset

ds = load_dataset("Ekimetrics/climateqa-questions-3k-1.0", split='train')
french = 0
english = 0
# Loop through all questions in the dataset
for item in ds:
    if 'é' in item['question']:
        french += 1
    else:
        english += 1
        print(item['question'])

print(french, english)


