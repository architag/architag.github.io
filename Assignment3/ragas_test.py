# Step 1: Install ragas (run in terminal)
# pip install ragas

# Step 2: Prepare your data
from datasets import Dataset

data = {
    "question": [
        "What is climate change?",
        "How does CO2 affect global warming?"
    ],
    "answer": [
        "Climate change refers to long-term shifts in temperatures and weather patterns.",
        "CO2 traps heat in the atmosphere, leading to global warming."
    ],
    "contexts": [
        ["Climate change refers to long-term shifts in temperatures and weather patterns."],
        ["CO2 is a greenhouse gas that traps heat in the atmosphere."]
    ]
}

ds = Dataset.from_dict(data)

# Step 3: Import metrics
OPEN_AI_KEY="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"
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference

metrics = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextPrecisionWithoutReference()
]

# Step 4: Evaluate
from ragas import evaluate

results = evaluate(ds, metrics=metrics)
print(results)

# Step 5: Optional - convert to DataFrame
df = results.to_pandas()
print(df.head())
