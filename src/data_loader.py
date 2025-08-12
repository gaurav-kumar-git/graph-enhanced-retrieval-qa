import json
from typing import List, Dict, Set, Any

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Loads the full dataset from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _process_passages_to_map(context: List[list]) -> Dict[str, str]:
    """
    CHANGE: Processes raw context into a dictionary mapping title to passage text.
    This is safer than returning two parallel lists.

    Args:
        context: The 'context' field from a single data sample.

    Returns:
        A dictionary where keys are passage titles and values are the full passage texts.
    """
    passages_map = {}
    for title, sentences in context:
        # Check if sentences list is not empty
        if sentences:
            # Join sentences and store in the map with the title as the key
            passages_map[title] = " ".join(sentences)
    return passages_map


def _get_ground_truth_titles(supporting_facts: List[list]) -> Set[str]:
    """
    CHANGE: Renamed to be a "private" helper. Functionality is the same.
    Extracts the unique ground truth titles from the supporting_facts field.

    Args:
        supporting_facts: The 'supporting_facts' field from a single sample.

    Returns:
        A set of unique titles that are the ground truth for retrieval.
    """
    return {title for title, sent_idx in supporting_facts}


def process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    NEW: A single, convenient function to process one raw data sample.
    This is the main function you will call from other parts of your project.

    Args:
        sample: A single dictionary element from the loaded dataset.

    Returns:
        A clean dictionary containing the essential processed fields.
        Example:
        {
            'question': 'Who is the mother of ...?',
            'passages': {'title1': 'text1...', 'title2': 'text2...'},
            'ground_truth_titles': {'title1', 'title3'}
        }
    """
    question = sample['question']
    context = sample['context']
    supporting_facts = sample['supporting_facts']

    processed_passages = _process_passages_to_map(context)
    ground_truth_titles = _get_ground_truth_titles(supporting_facts)

    return {
        'question': question,
        'passages': processed_passages,
        'ground_truth_titles': ground_truth_titles,
    }