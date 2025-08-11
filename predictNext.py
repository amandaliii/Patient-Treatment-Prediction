import torch
from info import CATEGORY_TO_D_ITEMS

# predict the next item from a given input sequence and check if prediction matches ground truth
def predict_next(model, input_seq, ground_truth, item2idx, idx2item, max_len=20, category=None, itemid_label_mappings_flat=None):
    # evaluation mode
    model.eval()
    # get current device (cpu/gpu)
    device = next(model.parameters()).device
    # convert input sequence items to strings to match itemid_label_mappings
    # map input sequence items to indices
    input_ids = [item2idx.get(str(i), item2idx['<UNK>']) for i in input_seq]
    # pad if needed to max_len with PAD (index 0)
    if len(input_ids) < max_len:
        input_ids = [0] * (max_len - len(input_ids)) + input_ids
    # keep only max_len most recent tokens for context
    input_ids = input_ids[-max_len:]
    # create tensor with batch size 1
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        # forward pass
        logits = model(input_tensor)
        # select most probable next item index
        pred_id = logits.argmax(dim=-1).item()
        # convert predicted index to item string
        predicted_item = idx2item[pred_id]

    # get labels for predicted and ground truth ITEMIDs
    d_items_category = CATEGORY_TO_D_ITEMS.get(category, 'Uncategorized')
    # debug: print lookup details
    print(f"\nPredicting for Category: {category}, D_ITEMS Category: {d_items_category}")
    print(f"  Ground Truth ITEMID: {ground_truth}, Predicted ITEMID: {predicted_item}")

    # lookup labels from the flat mapping (fallback to 'Unknown')
    if category in ('prescriptions', 'microbiology_events'):
        predicted_label = predicted_item
        ground_truth_label = ground_truth
    else:
        predicted_label = itemid_label_mappings_flat.get(str(predicted_item), 'Unknown')
        ground_truth_label = itemid_label_mappings_flat.get(str(ground_truth), 'Unknown')

    print(f"\nPredicted ITEMID: {predicted_item}, Label: {predicted_label}")
    print(f"Ground Truth ITEMID: {ground_truth}, Label: {ground_truth_label}")

    return predicted_item, predicted_label, ground_truth_label, str(predicted_item) == str(ground_truth)

