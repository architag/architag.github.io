from bert_score import score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, torch.nn.functional as F
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
import pandas as pd
import csv

def load_reference_questions():
    """
    Load reference questions from the CSV file.
    
    Returns:
        list: List of dictionaries with reference context and question
    """
    eval_data = []
    try:
        with open('reference_questions.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                eval_data.append({
                    "ref": row["Reference Context"],
                    "ques": row["Question"]
                })
    except Exception as e:
        print(f"Error loading reference questions: {e}")
        return []
    
    return eval_data

def compute_lexical_f1(generated, reference):
    """
    Compute lexical F1 score between generated and reference text.
    
    Args:
        generated (str): Generated answer text
        reference (str): Reference answer text
        
    Returns:
        float: F1 score
    """
    import re
    import string
    
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[' + string.punctuation + ']', ' ', text)
        return set(text.split())
    
    gen_tokens = normalize_text(generated)
    ref_tokens = normalize_text(reference)
    common_tokens = gen_tokens.intersection(ref_tokens)
    
    if len(gen_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(gen_tokens) if len(gen_tokens) > 0 else 0
    recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_semantic_similarity(generated, reference):
    """
    Compute semantic similarity between generated and reference text using embeddings.
    
    Args:
        generated (str): Generated answer text
        reference (str): Reference answer text
        
    Returns:
        float: Semantic similarity score (cosine similarity)
    """
    embeddings = HuggingFaceEmbeddings(model_name="climatebert/distilroberta-base-climate-f")
    gen_embedding = embeddings.embed_query(generated)
    ref_embedding = embeddings.embed_query(reference)
    gen_embedding = np.array(gen_embedding)
    ref_embedding = np.array(ref_embedding)
    similarity = np.dot(gen_embedding, ref_embedding) / (np.linalg.norm(gen_embedding) * np.linalg.norm(ref_embedding))
    return float(similarity)

def compute_entailment_score(generated, reference):
    """
    Compute entailment score between generated and reference text.
    
    Args:
        generated (str): Generated answer text
        reference (str): Reference answer text
        
    Returns:
        float: Entailment score
    """
    model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(reference, generated, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        entailment_score = scores[0, 2].item()
        
    return entailment_score

def evaluate_causal_score(eval_data):
    for eval_item in eval_data:
        scores = {
            "lexical_f1": compute_lexical_f1(eval_item["pred"], eval_item["ref"]),
            "semantic_similarity": compute_semantic_similarity(eval_item["pred"], eval_item["ref"]),
            "entailment": compute_entailment_score(eval_item["pred"], eval_item["ref"])
        }
        scores["combined_score"] = (
            scores["lexical_f1"] * 0.3 + 
            scores["semantic_similarity"] * 0.4 + 
            scores["entailment"] * 0.3
        )
        eval_item["scores"] = scores

def write_to_csv(eval_data):
    with open('evaluation_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ref', 'ques', 'pred', 'lexical_f1_score', 'semantic_similarity_score', 
                     'entailment_score', 'combined_score']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in eval_data:
            row = {
                'ref': item['ref'],
                'ques': item['ques'],
                'pred': item['pred'],
                'lexical_f1_score': round(item['scores']['lexical_f1'], 3), 
                'semantic_similarity_score': round(item['scores']['semantic_similarity'], 3),
                'entailment_score': round(item['scores']['entailment'], 3),
                'combined_score': round(item['scores']['combined_score'], 3)
            }
            writer.writerow(row)
    print(f"Evaluation results written to evaluation_results.csv")

def main():
    eval_data = load_reference_questions()
    evaluate_causal_score(eval_data)
    write_to_csv(eval_data)

if __name__ == "__main__":
    main()