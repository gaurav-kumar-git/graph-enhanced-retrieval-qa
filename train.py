import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os 
import glob 

from src.models.gnn import GNNRanker
from src.engine import train_one_epoch

EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
MODEL_PATH_BGE = '/home/sslab/24m0786/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
PROCESSED_TRAIN_DATA_DIR = 'data/processed/train_graphs/'
SAVE_MODEL_PATH = 'saved_models/gnn_ranker_final.pt'

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    sentence_model = SentenceTransformer(MODEL_PATH_BGE, device=device)
    embedding_dim = sentence_model.get_sentence_embedding_dimension()
    
    gnn_model = GNNRanker(
        in_channels=embedding_dim,
        hidden_channels=512,
        out_channels=embedding_dim
    ).to(device)

    print(f"Loading pre-processed graph data from '{PROCESSED_TRAIN_DATA_DIR}'...")
    
    graph_files = glob.glob(os.path.join(PROCESSED_TRAIN_DATA_DIR, '*.pt'))
    
    if not graph_files:
        print(f"ERROR: No pre-processed graph files found in '{PROCESSED_TRAIN_DATA_DIR}'.")
        print("Please run 'preprocess.py' first.")
        return
    
    train_data_list = [torch.load(f, weights_only=False) for f in tqdm(graph_files, desc="Loading Graphs")]
    print(f"Loaded {len(train_data_list)} graphs.")

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(gnn_model, sentence_model, train_data_list, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")

    torch.save(gnn_model.state_dict(), SAVE_MODEL_PATH)
    print(f"\nTraining complete. Model saved to {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    main()