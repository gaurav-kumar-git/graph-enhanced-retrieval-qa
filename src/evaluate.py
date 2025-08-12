from typing import List, Set

def calculate_mrr(ranked_titles: List[str], ground_truth_titles: Set[str]) -> float:
    """Calculates the Mean Reciprocal Rank for a single sample."""
    for i, title in enumerate(ranked_titles):
        if title in ground_truth_titles:
            return 1.0 / (i + 1)
    return 0.0

def calculate_f1(ranked_titles: List[str], ground_truth_titles: Set[str]) -> float:
    """Calculates the F1 score for a single sample."""
    if not ground_truth_titles:
        return 0.0
    
    k = len(ground_truth_titles)
    retrieved_top_k = set(ranked_titles[:k])
    
    num_correct = len(retrieved_top_k.intersection(ground_truth_titles))
    
    precision = num_correct / k if k > 0 else 0.0
    recall = num_correct / len(ground_truth_titles)
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1