import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List

def construct_graph(
    passages_map: Dict[str, str], 
    model: SentenceTransformer, 
    similarity_threshold: float = 0.8
) -> Data:
    """
    Constructs a Graph of Passages (GoP) for a single sample.

    Args:
        passages_map: A dictionary mapping passage titles to passage texts.
        model: A pre-loaded SentenceTransformer model for encoding.
        similarity_threshold: The threshold for creating semantic edges.

    Returns:
        A PyTorch Geometric Data object representing the graph.
    """
    passage_titles = list(passages_map.keys())
    passage_texts = list(passages_map.values())
    
    if not passage_texts:
        # Return an empty graph if there are no passages
        return Data(x=torch.empty((0, model.get_sentence_embedding_dimension())), 
                    edge_index=torch.empty((2, 0), dtype=torch.long))

    # 1. Node Feature Generation
    node_features = model.encode(
        passage_texts,
        convert_to_tensor=True,
        batch_size=32
    ).to(model.device)

    # 2. Edge Heuristic 1: Semantic Similarity
    similarity_matrix = util.cos_sim(node_features, node_features)
    edge_indices_similarity = torch.where(similarity_matrix > similarity_threshold)
    similarity_edges = []
    for i in range(len(edge_indices_similarity[0])):
        u, v = edge_indices_similarity[0][i].item(), edge_indices_similarity[1][i].item()
        if u != v:
            similarity_edges.append((u, v))

    # 3. Edge Heuristic 2: Sequential
    num_passages = len(passage_titles)
    sequential_edges = []
    for i in range(num_passages - 1):
        sequential_edges.append((i, i + 1))
        sequential_edges.append((i + 1, i))

    # 4. Combine Edges & Create edge_index
    all_edges = similarity_edges + sequential_edges
    if not all_edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        unique_edges = list(set(all_edges))
        edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()

    # 5. Assemble and return the final graph object
    graph_data = Data(x=node_features, edge_index=edge_index)
    
    return graph_data