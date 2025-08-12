import torch
from tqdm import tqdm

def train_one_epoch(gnn_model, sentence_model, data_list, optimizer, loss_fn, device):
    """
    Performs one full epoch of training.
    """
    gnn_model.train()  # Set the GNN model to training mode
    total_loss = 0

    for data in tqdm(data_list, desc="Training Epoch"):
        optimizer.zero_grad()

        # Unpack data and move to the correct device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        question = data.question
        positive_indices = data.positive_node_indices.to(device)
        
        # We need at least one positive and one negative sample to compute triplet loss
        if len(positive_indices) == 0 or len(positive_indices) == len(x):
            continue

        # 1. Get updated passage embeddings from the GNN
        updated_passage_embeddings = gnn_model(x, edge_index)
        
        # 2. Get the question embedding (anchor)
        # We encode the question on-the-fly
        anchor_embedding = sentence_model.encode(question, convert_to_tensor=True).to(device)

        # 3. Get positive and negative embeddings
        positive_embeddings = updated_passage_embeddings[positive_indices]
        
        # Create a mask to find negative indices
        full_indices = torch.arange(len(x), device=device)
        is_positive = torch.isin(full_indices, positive_indices)
        negative_indices = full_indices[~is_positive]
        
        negative_embeddings = updated_passage_embeddings[negative_indices]

        # 4. Calculate Triplet Loss
        # We need to expand the tensors to make them broadcastable for the loss function
        # Anchor: [1, D], Positive: [P, D], Negative: [N, D]
        # Loss wants [B, D], [B, D], [B, D] where B is batch size
        # We'll compare every negative to every positive
        num_pos = len(positive_embeddings)
        num_neg = len(negative_embeddings)
        
        anchor = anchor_embedding.unsqueeze(0).expand(num_pos * num_neg, -1)
        pos = positive_embeddings.repeat_interleave(num_neg, dim=0)
        neg = negative_embeddings.repeat(num_pos, 1)

        loss = loss_fn(anchor, pos, neg)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(data_list)