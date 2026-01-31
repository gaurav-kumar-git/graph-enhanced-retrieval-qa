import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import json

from src.data_loader import load_dataset, process_sample
from src.graph_constructor import construct_graph

MODEL_PATH_BGE = '/home/sslab/24m0786/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
RAW_DATA_PATH = 'data/raw/train.json'
PROCESSED_GRAPHS_DIR = 'data/processed/train_graphs/' 

def main():
    """
    This script preprocesses the raw data incrementally.
    It constructs a graph for each sample and saves it as an individual file.
    If a graph file already exists, it skips the sample, making the process resumable.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    sentence_model = SentenceTransformer(MODEL_PATH_BGE, device=device)

    print(f"Loading raw data from {RAW_DATA_PATH}...")
    raw_train_data = load_dataset(RAW_DATA_PATH)
    
    os.makedirs(PROCESSED_GRAPHS_DIR, exist_ok=True)
    
    print("Starting incremental graph construction...")
    
    # We will use a unique ID for each sample. The '_id' field is perfect for this.
    # If it doesn't exist, we can create a hash of the question as a fallback.
    for i, sample in enumerate(tqdm(raw_train_data, desc="Constructing Graphs")):
        
        # 1. Create a unique filename for this sample's graph.
        sample_id = sample.get('_id', None)
        if sample_id is None:
             # Fallback if '_id' is not present: create a simple hash
             sample_id = f"sample_{i}_{hash(sample['question'])}"

        graph_filename = os.path.join(PROCESSED_GRAPHS_DIR, f"{sample_id}.pt")
        
        # 2. Check if the graph file already exists. If so, skip.
        if os.path.exists(graph_filename):
            continue # Skip to the next sample

        # If the graph doesn't exist, process the sample
        processed = process_sample(sample)
        
        if not processed['passages']:
            continue

        # Construct the graph
        graph = construct_graph(processed['passages'], sentence_model)
        
        # Attach necessary metadata for training
        graph.question = processed['question']
        passage_titles = list(processed['passages'].keys())
        positive_indices = [i for i, title in enumerate(passage_titles) if title in processed['ground_truth_titles']]
        graph.positive_node_indices = torch.tensor(positive_indices, dtype=torch.long)
        
        # 3. Save the single graph object to its unique file
        torch.save(graph, graph_filename)

    print(f"\nPreprocessing complete.")
    print(f"All graphs are saved in '{PROCESSED_GRAPHS_DIR}'.")

if __name__ == "__main__":
    main()