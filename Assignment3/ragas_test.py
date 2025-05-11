# local_ragas_like_eval.py

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# 1. Load local models
nli_model_name = "microsoft/deberta-xlarge-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Demo data
data = [
    {
        "question": "How does deforestation contribute to climate change?",
        "contexts": [
            "Deforestation reduces the amount of CO2 absorbed by trees, leading to higher atmospheric concentrations of greenhouse gases."
        ],
        "answer": "Deforestation increases CO2 levels in the atmosphere because trees that absorb carbon are removed."
    },
    {
        "question": "What role do aerosols play in climate cooling?",
        "contexts": [
            "Aerosols scatter sunlight and can lead to a net cooling effect by increasing the Earth's albedo."
        ],
        "answer": "Aerosols help cool the planet by scattering sunlight and increasing reflectivity."
    }
]

# 3. Define faithfulness scorer using NLI
def compute_faithfulness(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(nli_model.device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    entailment_score = probs[0][2].item()  # index 2 = entailment
    return entailment_score

# 4. Define answer relevancy using cosine similarity
def compute_relevance(question, answer):
    embeddings = embedder.encode([question, answer], convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

# 5. Evaluate
results = []
for row in data:
    context = " ".join(row["contexts"])
    answer = row["answer"]
    question = row["question"]

    faith = compute_faithfulness(context, answer)
    rel = compute_relevance(question, answer)

    results.append({
        "question": question,
        "faithfulness": round(faith, 3),
        "answer_relevance": round(rel, 3),
    })

# 6. Print results
df = pd.DataFrame(results)
print("\n=== Local RAG Evaluation Results ===")
print(df.to_markdown(index=False))