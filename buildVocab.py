from collections import Counter
import torch
import pandas as pd
from config import result

def build_vocab(sequences):
    """
    Build vocabulary from a list of sequences (each sequence is a list of items).

    Args:
        sequences (List[List[str|int]]): List of sequences with items (converted to strings inside)

    Returns:
        item2idx (dict): Maps item string → index.
        idx2item (dict): Reverse mapping index → item string.
        class_weights (torch.Tensor): Inverse frequency class weights for loss.
    """
    item_counts = Counter()
    for seq in sequences:
        # Count string versions of all items
        item_counts.update(str(item) for item in seq)

    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    vocab_items = special_tokens + [item for item, _ in item_counts.most_common()]

    item2idx = {item: idx for idx, item in enumerate(vocab_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}

    total_count = sum(item_counts.values())
    weights = []
    for item in vocab_items:
        if item in special_tokens:
            weights.append(1.0)  # equal weight for special tokens
        else:
            count = item_counts.get(item, 1)
            weight = total_count / (len(item_counts) * count)
            weights.append(weight)

    class_weights = torch.tensor(weights, dtype=torch.float)

    return item2idx, idx2item, class_weights

def save_vocab_to_excel(item2idx, class_weights=None, filepath='vocab.xlsx'):
    """
    Save the vocabulary and optionally the class weights to an Excel file.

    Args:
        item2idx (dict): Mapping from item string to index.
        class_weights (torch.Tensor, optional): Tensor of class weights.
        filepath (str): Excel file path to save to.
    """
    # Prepare data for DataFrame: item, index, and optionally weight
    data = []
    for item, idx in item2idx.items():
        weight = None
        if class_weights is not None and idx < len(class_weights):
            weight = class_weights[idx].item()  # convert tensor to python float
        data.append({'Item': item, 'Index': idx, 'Weight': weight})

    df = pd.DataFrame(data)

    # Save to Excel
    df.to_excel(filepath, index=False)
    print(f"Vocabulary saved to Excel file: {filepath}")

def load_vocab_from_excel(filepath='vocab.xlsx'):
    """
    Load vocabulary and class weights from an Excel file saved by save_vocab_to_excel.

    Returns:
        item2idx (dict): Mapping from item string to index.
        class_weights (torch.Tensor or None): Tensor of class weights if present, else None.
    """
    df = pd.read_excel(filepath)
    item2idx = dict(zip(df['Item'], df['Index']))
    if 'Weight' in df.columns:
        weights_list = df['Weight'].fillna(1.0).tolist()  # fill missing with 1.0
        class_weights = torch.tensor(weights_list, dtype=torch.float)
    else:
        class_weights = None
    return item2idx, class_weights

# extract all sequences aggregated per admission for vocab building
all_sequences = []
for hadm_id, data in result.items():
    seq = []
    for cat in ['chart_items', 'input_items', 'lab_items',
                'microbiology_items', 'prescriptions_items', 'procedure_items']:
        seq.extend(str(item) for item in data[cat])
    if seq:
        all_sequences.append(seq)

print(f"Total sequences to build vocab from: {len(all_sequences)}")

# build vocabulary from sequences
item2idx, idx2item, class_weights = build_vocab(all_sequences)
print(f"Vocabulary size: {len(item2idx)}")

# save vocab to Excel file
save_vocab_to_excel(item2idx, class_weights, filepath='mimic_vocab.xlsx')
