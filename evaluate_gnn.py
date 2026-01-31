import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np

from src.data_loader import load_dataset, process_sample
from src.graph_constructor import construct_graph
from src.models.gnn import GNNRanker
from src.evaluate import calculate_mrr, calculate_f1

MODEL_PATH_BGE = '/home/sslab/24m0786/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
SAVED_GNN_MODEL_PATH = 'saved_models/gnn_ranker_final.pt'
DEV_DATA_PATH = 'data/raw/dev.json'

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    sentence_model = SentenceTransformer(MODEL_PATH_BGE, device=device)
    embedding_dim = sentence_model.get_sentence_embedding_dimension()
    
    gnn_model = GNNRanker(embedding_dim, 512, embedding_dim).to(device)
    gnn_model.load_state_dict(torch.load(SAVED_GNN_MODEL_PATH))
    gnn_model.eval() 
    
    dev_dataset = load_dataset(DEV_DATA_PATH)
    
    gnn_mrrs, gnn_f1s = [], []
    with torch.no_grad(): 
        for sample in tqdm(dev_dataset, desc="Evaluating GNN"):
            processed = process_sample(sample)
            if not processed['passages']:
                continue

            # a. Construct graph
            graph = construct_graph(processed['passages'], sentence_model)
            x, edge_index = graph.x.to(device), graph.edge_index.to(device)
            
            # b. Get updated embeddings from GNN
            updated_passage_embeddings = gnn_model(x, edge_index)

            # c. Rank using new embeddings
            question_embedding = sentence_model.encode(processed['question'], convert_to_tensor=True).to(device)
            cosine_scores = util.cos_sim(question_embedding, updated_passage_embeddings)[0]
            
            top_results = torch.topk(cosine_scores, k=len(x))
            passage_titles = list(processed['passages'].keys())
            ranked_titles = [passage_titles[i] for i in top_results.indices]
            
            # d. Calculate metrics
            gnn_mrrs.append(calculate_mrr(ranked_titles, processed['ground_truth_titles']))
            gnn_f1s.append(calculate_f1(ranked_titles, processed['ground_truth_titles']))

    print("\n--- GNN Model Performance ---")
    print(f"  - Average MRR: {np.mean(gnn_mrrs):.4f}")
    print(f"  - Average F1:  {np.mean(gnn_f1s):.4f}")

if __name__ == "__main__":
    main()