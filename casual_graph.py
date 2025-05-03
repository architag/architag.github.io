import nltk
from nltk.tokenize import sent_tokenize
import logging
import argparse
import glob
from frame_semantic_transformer import FrameSemanticTransformer
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm

# Define causal trigger phrases (from FrameNet / fig S3B)
causal_triggers = {
    "cause", "causes", "caused", "lead to", "leads to", "led to",
    "bring about", "brought about", "result in", "resulted in",
    "contribute to", "contributed to", "trigger", "triggered"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_and_sentence_tokenize(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            text = text.replace('\n', ' ')
            sentences = sent_tokenize(text)
            return sentences
    except FileNotFoundError:
        logger.error(f"Error: File '{filepath}' not found.")
        return []
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return []

# def sentence_has_causal_trigger(sentence):
#     sentence_lower = sentence.lower()
#     return any(trigger in sentence_lower for trigger in causal_triggers)

lemmatizer = WordNetLemmatizer()

# Causal trigger set
causal_triggers = set([
    "because of.prep", "because.c", "bring about.v", "bring on.v", "bring.v",
    "causative.a", "cause.n", "cause.v", "consequence.n", "consequent.a", "consequential.a",
    "dictate.v", "due to.prep", "for.c", "force.v", "give rise.v", "induce.v",
    "lead (to).v", "leave.v", "legacy.n", "make.v", "mean.v", "motivate.v",
    "precipitate.v", "put.v", "raise.v", "reason.n", "render.v", "responsible.a",
    "result (in).v", "result.n", "resultant.a", "resulting.a", "see.v", "send.v",
    "since.c", "so.c", "sway.v", "wreak.v"
])

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def sentence_has_causal_trigger(sentence, trigger_location):
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    
    if trigger_location < 0 or trigger_location >= len(words):
        return False
    
    word, pos = pos_tags[trigger_location]
    wn_pos = get_wordnet_pos(pos)
    if wn_pos is None:
        return False

    lemma = lemmatizer.lemmatize(word.lower(), wn_pos)
    tag = f"{lemma}.{wn_pos[0]}"  # e.g., cause.v

    for trigger in causal_triggers:
        if tag in trigger or trigger.startswith(lemma):  # include multiword like "lead (to).v"
            return True
    return False

def process_files(files):
    frame_transformer = FrameSemanticTransformer(batch_size=64, use_gpu=True)

    all_step1_rows = []
    all_step3_rows = []

    for file in files[:10]:
        sentences = read_and_sentence_tokenize(file)
        if not sentences:
            continue

        logger.info(f"{len(sentences)} total sentences in {file}")
        # results = []
        # for sentence in tqdm(sentences):
        #     # logger.info(f"Processing sentence: {sentence}")
        #     # Detect frames in the sentence
        #     results.append(frame_transformer.detect_frames(sentence))
        results = frame_transformer.detect_frames_bulk(sentences)

        for res in results:
            for frame in res.frames:
                causes = [fe.text for fe in frame.frame_elements if fe.name == 'Cause']
                effects = [fe.text for fe in frame.frame_elements if fe.name == 'Effect']

                if causes and effects:
                    for cause in causes:
                        for effect in effects:
                            row = {
                                'sentence': res.sentence,
                                'frame_name': frame.name,
                                'cause': cause,
                                'effect': effect,
                                'file': file
                            }
                            all_step1_rows.append(row)
                            if sentence_has_causal_trigger(res.sentence):
                                all_step3_rows.append(row)

    # Create DataFrames
    df_step1 = pd.DataFrame(all_step1_rows)
    df_step3 = pd.DataFrame(all_step3_rows)

    # Save CSVs
    df_step1.to_csv("step1_cause_effect.csv", index=False)
    df_step3.to_csv("step3_with_causal_trigger.csv", index=False)

    # Save JSONs
    # df_step1.to_json("step1_cause_effect.json", orient='records', indent=2)
    # df_step3.to_json("step3_with_causal_trigger.json", orient='records', indent=2)

    logger.info(f"Saved {len(df_step1)} rows after Step 1 and {len(df_step3)} after Step 3.")


def main():
    parser = argparse.ArgumentParser(description="Process text files and extract sentences.")
    parser.add_argument('--data_dir', type=str, help='Directory of text files to process')
    args = parser.parse_args()

    files = glob.glob(f"{args.data_dir}/**/*.txt", recursive=True)
    logger.info(f"Found {len(files)} files in {args.data_dir}")
    process_files(files)

if __name__ == "__main__":
    main()
