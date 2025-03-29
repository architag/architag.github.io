import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import argparse
import os


# Parse arguments
parser = argparse.ArgumentParser(description="Knowledge Graph Generator")
parser.add_argument("--output_dir", type=str, default="./", help="Directory to save outputs")
args = parser.parse_args()

OUTPUT_DIR = args.output_dir

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


# Generate and visualize the knowledge graph
print("Generating knowledge graph...")
triples = [
    ("Droughts", "are", "natural hazard"),
    ("Droughts", "have", "cascading impacts"),
    ("Cascading impacts", "affect", "economic sectors"),
    ("Cascading impacts", "affect", "environment"),
    ("Cascading impacts", "affect", "society"),
    ("Droughts", "lead to", "agriculture production losses"),
    ("Droughts", "lead to", "intense wildfires"),
    ("Droughts", "lead to", "waterways disruptions"),
    ("Droughts", "lead to", "water supply shortages"),
    ("Improved drought forecasts", "help", "deal with impacts"),
    ("Accurate forecasting of drought", "is", "a challenge"),
    ("Climate change", "compounds", "forecasting challenge"),
    ("Drought indices", "are used to", "monitor droughts"),
    ("Drought indices", "are used to", "quantify droughts"),
    ("Several drought indices", "have been proposed", "with different complexities"),
    ("Standardized Precipitation Index (SPI)", "is", "a drought index"),
    ("Standardized Precipitation Evapo-Transpiration Index (SPEI)", "is", "a drought index"),
    ("SPEI", "takes into account", "atmospheric water balance"),
    ("SPEI", "is suited for", "climate change context"),
    ("Several approaches", "have been proposed", "to forecast SPEI"),
    ("Approaches", "include", "stochastic techniques"),
    ("Approaches", "include", "probabilistic techniques"),
    ("Approaches", "include", "machine learning techniques"),
    ("Artificial Neural Network (ANN)", "is used for", "drought forecasting"),
    ("Long Short-Term Memory (LSTM)", "is used for", "drought forecasting"),
    ("Convolutional LSTM", "is used for", "drought forecasting"),
    ("Wavelet ANN", "is used for", "drought forecasting"),
    ("Integrated ANN", "is used for", "drought forecasting"),
    ("Hybrid neural network", "combines", "multiple models"),
    ("Hybrid neural network", "is trained with", "different losses"),
    ("Hybrid neural network", "improves", "drought forecasting accuracy"),
    ("Existing work", "does not emphasize", "evaluation of extreme drought"),
    ("Existing work", "does not emphasize", "analysis of extreme drought"),
    ("Existing work", "does not emphasize", "evaluation of severe wet events"),
    ("Existing work", "does not emphasize", "analysis of severe wet events"),
    ("Imbalance in time-series", "makes difficult", "forecasting extreme events"),
    ("Extremely dry events", "are", "few"),
    ("Extremely wet events", "are", "few"),
    ("Few extreme events", "make difficult", "forecasting"),
    ("This paper", "develops", "SQUASH loss function"),
    ("SQUASH loss function", "is", "differentiable"),
    ("SQUASH loss function", "captures", "shape error"),
    ("SQUASH loss function", "handles", "imbalanced data"),
    ("SQUASH loss function", "is", "computationally efficient"),
    ("This paper", "validates", "multi-step SPEI forecasting"),
    ("Validation", "is performed on", "USA"),
    ("Validation", "is performed on", "India"),
    ("This article", "presents", "ablation study"),
    ("Ablation study", "includes", "different surrogate loss functions")
]

knowledge_graph = nx.DiGraph()
for subject, relation, obj in triples:
    knowledge_graph.add_edge(subject, obj, label=relation)

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
