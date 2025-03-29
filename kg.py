import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import spacy
from tqdm import tqdm
import argparse
import os


# Parse arguments
parser = argparse.ArgumentParser(description="Knowledge Graph Generator")
parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
parser.add_argument("--output_dir", type=str, default="./", help="Directory to save outputs")
args = parser.parse_args()

FINETUNED_MODEL_PATH = args.model_path
OUTPUT_DIR = args.output_dir

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load spaCy for NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy English model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

def extract_entities_relations(text):
    """Extract entities and relations from text using spaCy"""
    doc = nlp(text)
    entities = set()
    relations = []
    
    # Extract named entities
    for ent in doc.ents:
        entities.add((ent.text, ent.label_))
    
    # Extract noun chunks and verbs for relations
    for token in doc:
        if token.dep_ in ("nsubj", "dobj", "pobj"):
            if token.head.text.lower() not in ["is", "are", "was", "were"]:
                relations.append((token.text, token.head.text, token.dep_))
    
    return entities, relations

def generate_knowledge_graph(prompts, num_samples=5, max_length=500):
    """Generate knowledge graph from model outputs"""
    G = nx.DiGraph()
    
    for prompt in tqdm(prompts[:num_samples], desc="Generating knowledge graph"):
        try:
            # Generate text from the model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract entities and relations
            entities, relations = extract_entities_relations(generated_text)
            
            # Add to graph
            for entity, label in entities:
                if not G.has_node(entity):
                    G.add_node(entity, label=label, type="entity")
                
            for src, dst, rel in relations:
                if src and dst:  # Ensure neither is empty
                    if not G.has_node(src):
                        G.add_node(src)
                    if not G.has_node(dst):
                        G.add_node(dst)
                    G.add_edge(src, dst, label=rel)
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {str(e)}")
            continue
    
    return G

def visualize_graph(G, output_file="knowledge_graph.html"):
    """Visualize the knowledge graph using pyvis"""
    try:
        net = Network(height="800px", width="100%", directed=True, notebook=False)
        
        # Add nodes and edges
        for node in G.nodes():
            net.add_node(node, 
                         label=node, 
                         title=f"{node} ({G.nodes[node].get('label', 'entity')})",
                         group=G.nodes[node].get('label', 'entity'))
        
        for edge in G.edges():
            net.add_edge(edge[0], edge[1], 
                         label=G.edges[edge].get("label", ""),
                         title=G.edges[edge].get("label", ""))
        
        # Customize visualization
        net.toggle_physics(True)
        net.show_buttons(filter_=['physics'])
        
        # Ensure the output directory exists
        output_path = os.path.join(OUTPUT_DIR, output_file)
        net.save_graph(output_path)
        print(f"Knowledge graph saved to {output_path}")
    except Exception as e:
        print(f"Error visualizing graph: {str(e)}")

# Example prompts that might trigger domain-specific knowledge
# domain_prompts = [
#     "Climate change is caused by increasing concentrations of greenhouse gases like carbon dioxide and methane in the atmosphere.",
#     "Melting glaciers and polar ice caps contribute to rising sea levels, threatening coastal communities worldwide.",
#     "Climate change is causing shifts in plant and animal habitats, leading to biodiversity loss and species extinction.",
#     "Climate adaptation measures help communities prepare for the unavoidable impacts of climate change.",
#     "Climate justice emphasizes that vulnerable populations who contribute least to emissions often suffer the most severe impacts."
# ]

domain_prompts = ["Climate change is caused by an increase in greenhouse gases such as C02."]

# Generate and visualize the knowledge graph
print("Generating knowledge graph...")
knowledge_graph = generate_knowledge_graph(domain_prompts)

if len(knowledge_graph.nodes) > 0:
    print(f"Graph contains {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges")
    visualize_graph(knowledge_graph, "llama_knowledge_graph.html")
    
    # Save the graph as GEXF for further analysis
    gexf_path = os.path.join(OUTPUT_DIR, "llama_knowledge_graph.gexf")
    nx.write_gexf(knowledge_graph, gexf_path)
    print(f"Graph data saved to {gexf_path}")

    # Optional: Simple matplotlib visualization
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(knowledge_graph, k=0.5, iterations=50)
    nx.draw(knowledge_graph, pos, with_labels=True, node_size=2000, font_size=10)
    edge_labels = nx.get_edge_attributes(knowledge_graph, 'label')
    nx.draw_networkx_edge_labels(knowledge_graph, pos, edge_labels=edge_labels)
    png_path = os.path.join(OUTPUT_DIR, "llama_knowledge_graph.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static graph image saved to {png_path}")
else:
    print("No valid graph was generated - check your prompts and model outputs")
