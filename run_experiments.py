# In run_experiments.py (Corrected Version)
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import json
import time
import os
import glob

# Import your custom modules
from src.models.gnn_experimental import GNNExperimental
from src.engine import train_one_epoch
from src.data_loader import load_dataset, process_sample
from src.graph_constructor import construct_graph
from src.evaluate import calculate_mrr, calculate_f1

# --- CONFIGURATION ---
EXPERIMENT_EPOCHS = 5
LEARNING_RATE = 1e-4
MODEL_PATH_BGE = '/home/sslab/24m0786/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
# --- UPDATED PATH to be a directory ---
PROCESSED_TRAIN_DATA_DIR = 'data/processed/train_graphs/'
DEV_DATA_PATH = 'data/raw/dev.json'
RESULTS_FILE = 'results/experiment_results.json'

# run_single_experiment function remains the same as before...
def run_single_experiment(config, sentence_model, train_data, dev_data, device):
    print(f"\n{'='*20}\nRunning Experiment: {config}\n{'='*20}")
    embedding_dim = sentence_model.get_sentence_embedding_dimension()

    gnn_model = GNNExperimental(
        in_channels=embedding_dim, hidden_channels=512, out_channels=embedding_dim,
        num_layers=config['num_layers'], aggr=config['aggregation']
    ).to(device)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    start_time = time.time()
    for epoch in range(EXPERIMENT_EPOCHS):
        avg_loss = train_one_epoch(gnn_model, sentence_model, train_data, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{EXPERIMENT_EPOCHS} - Loss: {avg_loss:.4f}")
    training_time = time.time() - start_time

    gnn_model.eval()
    gnn_mrrs, gnn_f1s = [], []
    with torch.no_grad():
        for sample in tqdm(dev_data, desc=f"Evaluating {config}"):
            processed = process_sample(sample)
            if not processed['passages']: continue
            graph = construct_graph(processed['passages'], sentence_model)
            x, edge_index = graph.x.to(device), graph.edge_index.to(device)
            updated_embs = gnn_model(x, edge_index)
            q_emb = sentence_model.encode(processed['question'], convert_to_tensor=True).to(device)
            scores = util.cos_sim(q_emb, updated_embs)[0]
            top_results = torch.topk(scores, k=len(x))
            titles = list(processed['passages'].keys())
            ranked_titles = [titles[i] for i in top_results.indices]
            gnn_mrrs.append(calculate_mrr(ranked_titles, processed['ground_truth_titles']))
            gnn_f1s.append(calculate_f1(ranked_titles, processed['ground_truth_titles']))

    result = {
        'config': config, 'mrr': np.mean(gnn_mrrs), 'f1': np.mean(gnn_f1s),
        'training_time_seconds': training_time
    }
    return result

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    configs_to_run = [
        {'num_layers': 2, 'aggregation': 'mean'}, {'num_layers': 2, 'aggregation': 'max'},
        {'num_layers': 2, 'aggregation': 'sum'}, {'num_layers': 1, 'aggregation': 'mean'},
        {'num_layers': 3, 'aggregation': 'mean'}, {'num_layers': 4, 'aggregation': 'mean'},
    ]

    # --- CORRECTED DATA LOADING ---
    sentence_model = SentenceTransformer(MODEL_PATH_BGE, device=device)
    
    print(f"Loading pre-processed graph data from directory: '{PROCESSED_TRAIN_DATA_DIR}'...")
    graph_files = glob.glob(os.path.join(PROCESSED_TRAIN_DATA_DIR, '*.pt'))
    
    if not graph_files:
        print(f"ERROR: No pre-processed graph files found in '{PROCESSED_TRAIN_DATA_DIR}'.")
        print("Please run 'preprocess.py' first.")
        return

    train_data_list = [torch.load(f, weights_only=False) for f in tqdm(graph_files, desc="Loading Graphs")]
    print(f"Loaded {len(train_data_list)} graphs.")
    
    dev_dataset = load_dataset(DEV_DATA_PATH)

    all_results = []
    for config in configs_to_run:
        result = run_single_experiment(config, sentence_model, train_data_list, dev_dataset, device)
        all_results.append(result)
        print(f"Result for {config}: MRR={result['mrr']:.4f}, F1={result['f1']:.4f}")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll experiments complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()