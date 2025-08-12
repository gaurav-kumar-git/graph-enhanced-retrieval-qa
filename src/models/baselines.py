import torch
from rank_bm25 import BM25Okapi
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util

def run_bm25(question: str, passages_map: Dict[str, str]) -> List[str]:
    """
    Ranks passages against a question using BM25.
    (This function is fine and does not need changes)
    """
    corpus_titles = list(passages_map.keys())
    corpus_texts = list(passages_map.values())

    if not corpus_texts:
        return []

    tokenized_corpus = [doc.split(" ") for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = question.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    
    ranked_results = sorted(zip(corpus_titles, doc_scores), key=lambda x: x[1], reverse=True)
    ranked_titles = [title for title, score in ranked_results]
    
    return ranked_titles


def run_dpr(question: str, passages_map: Dict[str, str], model: SentenceTransformer) -> List[str]:
    """
    Ranks passages against a question using a dense retriever model.
    """
    corpus_titles = list(passages_map.keys())
    corpus_texts = list(passages_map.values())
    
    if not corpus_texts:
        return []

    question_embedding = model.encode(question, convert_to_tensor=True)
    
    passage_embeddings = model.encode(
        corpus_texts, 
        convert_to_tensor=True, 
        batch_size=32,
        show_progress_bar=False 
    )
    
    cosine_scores = util.cos_sim(question_embedding, passage_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=len(corpus_titles))
    ranked_titles = [corpus_titles[i] for i in top_results.indices]
    
    return ranked_titles