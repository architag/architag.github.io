import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from causal_score import RagScore
import csv

BASE_PATH = "/scratch/aa8716/mlsystems-spring-25"
llama_tokenizer = AutoTokenizer.from_pretrained(f"{BASE_PATH}/assignment_1/output_3/llama-epoch-1")
llama_model = AutoModelForCausalLM.from_pretrained(f"{BASE_PATH}/assignment_1/output_3/llama-epoch-1", device_map="auto")

def generate_answer(query, max_tokens=512):
    prompt = f"""
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
    eval_data = load_reference_questions()
    results = []
    for eval_item in eval_data:
        rag_result = generate_answer(eval_item["ques"])
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