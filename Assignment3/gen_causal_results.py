import torch
import time
import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
from causal_score import RagScore
import csv

BASE_PATH = "/scratch/aa8716/mlsystems-spring-25"
llama_tokenizer = AutoTokenizer.from_pretrained(f"{BASE_PATH}/assignment_1/Llama3.2-3B-hf")
llama_model = AutoModelForCausalLM.from_pretrained(f"{BASE_PATH}/assignment_1/Llama3.2-3B-hf", device_map="auto")

def load_vector_store(embed_model_name='bge', method='flat', device='cpu'):
    VECTOR_STORE_DIR = f"{BASE_PATH}/assignment_3/{method}/{embed_model_name}"
    if embed_model_name == 'climatebert':
        model_name = 'climatebert/distilroberta-base-climate-f'
    elif embed_model_name == 'minilm':
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    else:
        model_name = 'BAAI/bge-large-en'
    embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    vector_store = FAISS.load_local(folder_path=f"{VECTOR_STORE_DIR}/llama_vector_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully")
    return vector_store

def query_vector_store(vector_store, query_text, k=5):
    print(f"Searching for: '{query_text}'")
    results = vector_store.similarity_search(query_text, k=k)
    return [doc.page_content for doc in results]

def generate_answer(context, query, max_tokens=512):
    prompt = f"""
### Context:
{context}

### Question:
{query}

### Answer:
"""
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
    outputs = llama_model.generate(**inputs, max_new_tokens=max_tokens, eos_token_id=llama_tokenizer.eos_token_id, pad_token_id=llama_tokenizer.eos_token_id)
    answer = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_sequence = outputs[0]
    num_tokens_generated = generated_sequence.shape[0] - inputs['input_ids'].shape[1]
    print(f"Tokens generated: {num_tokens_generated}")
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    # Remove anything after the first occurrence of "###"
    if "###" in answer:
        answer = answer.split("###")[0].strip()
    return answer

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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vector_store = load_vector_store("bge", "flat", device)
    
    eval_data = load_reference_questions()
    results = []
    for eval_item in eval_data:
        retrieved_chunks = query_vector_store(vector_store, eval_item["ques"])
        context = "\n\n".join(retrieved_chunks)
        rag_result = generate_answer(context, eval_item["ques"])
        results.append({
            "ques": eval_item["ques"],
            "pred": rag_result,
            "ref": eval_item["ref"]
        })

    causal_score = RagScore()
    causal_score.evaluate_causal_score(results)
    causal_score.write_to_csv(results)

if __name__ == "__main__":
    main()