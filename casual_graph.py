# prompt: READ /content/02_SROCC_TS_FINAL.txt and create a list of senteces

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import logging
import argparse
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_and_sentence_tokenize(filepath):
  try:
    with open(filepath, 'r', encoding='utf-8') as file:  # Use utf-8 encoding
      text = file.read()
      # remove all \n from text
      text = text.replace('\n', ' ')
    sentences = sent_tokenize(text)
    return sentences
  except FileNotFoundError:
    logger.log(f"Error: File '{filepath}' not found.")
    return []
  except Exception as e:
    logger.log(f"An error occurred: {e}")
    return []

def process_files(files):
    for file in files:
        sentences = read_and_sentence_tokenize(file)
        if sentences:
            logger.info(f"{len(sentences)} total sentences in {file}")

def main():

    parser = argparse.ArgumentParser(description="Process text files and extract sentences.")
    parser.add_argument('--data_dir', type=str, help='Directory of text files to process')
    args = parser.parse_args()

    # Read all .txt files in the data_dir using glob
    files = glob.glob(f"{args.data_dir}/**/*.txt")
    logger.info(f"Found {len(files)} files in {args.data_dir}")
    process_files(files)

if __name__ == "__main__":
    main()
