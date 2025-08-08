# import dataset from dataprocessing file
from DataProcessing import load_mimic3_data
# used to count frequency of items (for building vocabulary)
from collections import Counter
# pytorch tools for tensors, dataset handling, and neural network layers
import torch
from itemID import create_itemid_label_mapping, load_labitems_labels

# load dataset from dataprocessing
mimic_data_dir = "/Users/amandali/Downloads/Mimic III"
# loads how many rows of mimic 3 data
result = load_mimic3_data(mimic_data_dir, nrows=2000000)
# load itemID to label mapping and check for duplicates (from D_ITEMS.csv)
itemid_label_mappings, duplicates = create_itemid_label_mapping(f"{mimic_data_dir}/D_ITEMS.csv")
# load lab item labels separately (from D_LABITEMS.csv)
lab_items_labels = load_labitems_labels(f"{mimic_data_dir}/D_LABITEMS.csv")

# merge all
if 'labevents' not in itemid_label_mappings:
    itemid_label_mappings['labevents'] = {}
for itemid, label in lab_items_labels.items():
    itemid_label_mappings['labevents'][itemid] = label

# debugging: print mapping details to verify content including merged lab labels
print("\nLoaded itemid_label_mappings (including lab items):")
for category, items in itemid_label_mappings.items():
    print(f"  Category: {category}, Number of ITEMIDs: {len(items)}")
    # Print a few sample ITEMIDs and labels for each category
    sample_items = list(items.items())[:3]
    for itemid, label in sample_items:
        print(f"    ITEMID: {itemid} -> Label: {label}")

# print duplicates if any
if duplicates:
    print("\nDuplicate ITEMIDs found in D_ITEMS.csv:")
    for dup in duplicates:
        print(f"  ITEMID: {dup['ITEMID']}")
        print(f"    First occurrence: Label='{dup['First_Label']}', Category='{dup['First_Category']}'")
        print(f"    Second occurrence: Label='{dup['Second_Label']}', Category='{dup['Second_Category']}'")
else:
    print("\nNo duplicate ITEMIDs found in D_ITEMS.csv.")

# category configuration maps event categories to their item keys in dataset
CATEGORIES = {
    'chart_events': 'chart_items',
    'input_events': 'input_items',
    'lab_events': 'lab_items',
    'microbiology_events': 'microbiology_items',
    'prescriptions': 'prescriptions_items',
    'procedure_events': 'procedure_items'
}

# map model categories to D_ITEMS.csv categories for label lookup
CATEGORY_TO_D_ITEMS = {
    'chartevents': 'CHART',
    'inputevents': 'INPUT',
    'labevents': 'LAB',
    'microbiologyevents': 'MICROBIOLOGY',
    'prescriptions': 'PRESCRIPTIONS',
    'procedureevents': 'PROCEDURE'
}

# combine all category mappings into a single ITEMID -> label dictionary
def flatten_itemid_label_mappings(itemid_label_mappings):
    combined_mapping = {}
    for category_dict in itemid_label_mappings.values():
        combined_mapping.update(category_dict)
    return combined_mapping

# extract sequences for a specific category with HADM_IDs
def extract_sequences(data, category_key):
    # will hold tuples (hadm_id, item sequence)
    sequence_list = []
    for hadm_id, category_dict in data.items():
        items = category_dict.get(category_key, [])
        # get the list of items for the given category in this hospital admission
        # ensure sequence has at least 2 items to be meaningful for prediction
        if len(items) >= 2:
            sequence_list.append((hadm_id, items))
    # return list of (hadm_id, items) tuples
    return sequence_list

# build vocabulary for a category
def build_vocab(sequences):
    # initialize counter for item frequencies
    item_counts = Counter()
    for seq in sequences:
        # convert items to strings
        item_counts.update(str(item) for item in seq)
        # update frequency counts with items from sequence
        item_counts.update(seq)
    # vocabulary list starting with special tokens for padding and unknown items, followed by items sorted by frequency
    vocab = ['<PAD>', '<UNK>'] + [item for item, _ in item_counts.most_common()]
    # map from item string to index
    item2idx = {item: i for i, item in enumerate(vocab)}
    # reverse mapping from index to item string
    idx2item = {i: item for item, i in item2idx.items()}

    # compute class weights for CrossEntropyLoss to handle imbalanced classes
    total = sum(item_counts.values())  # total number of item occurrences
    class_weights = torch.tensor(
        [total / (len(item_counts) * count) if count > 0 else 1.0 for item, count in item_counts.most_common()],
        dtype=torch.float)
    # weight inversely proportional to frequency for each item in vocab (excluding PAD and UNK)
    # weights for <PAD>, <UNK> tokens set as 1.0
    class_weights = torch.cat([torch.tensor([1.0, 1.0]), class_weights])
    # return vocab dicts and class weights tensor
    return item2idx, idx2item, class_weights

# flattened mapping of item IDs to label to itemid_label_mapping
flat_itemid_label_mapping = flatten_itemid_label_mappings(itemid_label_mappings)
