from bert_score import score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, torch.nn.functional as F
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import csv

class RagScore:

    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.entailment_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-base")
        self.entailment_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")

    def compute_entailment_score(self, generated, reference):
        """
        Compute entailment score between generated and reference text.
        
        Args:
            generated (str): Generated answer text
            reference (str): Reference answer text
            
        Returns:
            float: Entailment score
        """
        inputs = self.entailment_tokenizer(reference, generated, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.entailment_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            entailment_score = scores[0, 2].item()
            
        return entailment_score

    # def compute_semantic_similarity(self, generated, reference):
    #     """
    #     Compute semantic similarity between generated and reference text using embeddings.
        
    #     Args:
    #         generated (str): Generated answer text
    #         reference (str): Reference answer text
            
    #     Returns:
    #         float: Semantic similarity score (cosine similarity)
    #     """
    #     gen_embedding = self.semantic_similarity_embeddings.embed_query(generated)
    #     ref_embedding = self.semantic_similarity_embeddings.embed_query(reference)
    #     gen_embedding = np.array(gen_embedding)
    #     ref_embedding = np.array(ref_embedding)
    #     similarity = np.dot(gen_embedding, ref_embedding) / (np.linalg.norm(gen_embedding) * np.linalg.norm(ref_embedding))
    #     return float(similarity)
    def compute_relevance(self, question, answer):
        embeddings = self.embedder.encode([question, answer], convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    def compute_lexical_f1(self, generated, reference):
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

    def evaluate_causal_score(self, eval_data):
        for eval_item in eval_data:
            scores = {
                "lexical_f1": self.compute_lexical_f1(eval_item["pred"], eval_item["ref"]),
                "semantic_similarity": self.compute_relevance(eval_item["pred"], eval_item["ref"]),
                "entailment": self.compute_entailment_score(eval_item["pred"], eval_item["ref"])
            }
            scores["combined_score"] = (
                scores["lexical_f1"] * 0.3 + 
                scores["semantic_similarity"] * 0.4 + 
                scores["entailment"] * 0.3
            )
            eval_item["scores"] = scores

    def write_to_csv(self, eval_data):
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
