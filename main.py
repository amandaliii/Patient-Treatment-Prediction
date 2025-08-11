from info import MIMIC_DATA_DIR, NROWS, CATEGORIES
from dataProcessing import load_mimic3_data, load_vocab_from_excel
from itemIDToLabel import create_itemid_label_mapping, load_labitems_labels
from buildModel import SequenceDataset, decoderModel
from trainModel import train_model
from predictNext import predict_next

import torch
import pandas as pd
import random

def main():
    # Step 1: Load data
    print("Loading MIMIC-III data...")
    result = load_mimic3_data(MIMIC_DATA_DIR, nrows=NROWS)

    # Step 2: Load itemID-label mappings for interpretability
    print("Loading itemID-label mappings...")
    itemid_label_mappings = create_itemid_label_mapping(f"{MIMIC_DATA_DIR}/D_ITEMS.csv")
    lab_items_labels = load_labitems_labels(f"{MIMIC_DATA_DIR}/D_LABITEMS.csv")
    if 'labevents' not in itemid_label_mappings:
        itemid_label_mappings['labevents'] = {}
    for itemid, label in lab_items_labels.items():
        itemid_label_mappings['labevents'][itemid] = label

    # flatten mappings for quick lookup in predictions
    itemid_label_mappings = create_itemid_label_mapping(f"{MIMIC_DATA_DIR}/D_ITEMS.csv")
    lab_items_labels = load_labitems_labels(f"{MIMIC_DATA_DIR}/D_LABITEMS.csv")

    # merge lab items into main mapping
    itemid_label_mappings.update(lab_items_labels)

    # flat mapping is already done
    flat_itemid_label_mapping = itemid_label_mappings

    # Step 3: Prepare and train models for each category
    all_metrics = []
    all_prediction_rows = []
    NUM_RUNS = 3
    num_hadms = 10

    for category, cat_key in CATEGORIES.items():
        print(f"Processing category: {category}")
        # Extract sequences for this category
        sequence_tuples = [(hadm_id, data[cat_key]) for hadm_id, data in result.items() if len(data.get(cat_key, [])) >= 2]
        if not sequence_tuples:
            print(f"No sequences found for {category}, skipping.")
            continue

        # Shuffle and split into train/val sets
        random.seed(42)
        random.shuffle(sequence_tuples)
        train_size = int(0.8 * len(sequence_tuples))
        val_size = int(0.15 * len(sequence_tuples))
        train_seqs = [seq for _, seq in sequence_tuples[:train_size]]
        val_seqs = [seq for _, seq in sequence_tuples[train_size:train_size + val_size]]
        train_hadm_ids = [hadm_id for hadm_id, _ in sequence_tuples[:train_size]]
        val_hadm_ids = [hadm_id for hadm_id, _ in sequence_tuples[train_size:train_size + val_size]]

        # load vocab already done in dataProcessing
        item2idx, class_weights = load_vocab_from_excel('vocab.xlsx')
        idx2item = {idx: item for item, idx in item2idx.items()}

        # create datasets and dataloaders
        train_dataset = SequenceDataset(train_seqs, item2idx)
        val_dataset = SequenceDataset(val_seqs, item2idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

        # initialize model
        model = decoderModel(len(item2idx), embed_size=64, hidden_size=256)

        # train model
        metrics = train_model(model, train_loader, val_loader, class_weights, epochs=10)
        for m in metrics:
            m['Category'] = category
        all_metrics.extend(metrics)

        # save model
        torch.save(model.state_dict(), f"models/model_{category}.pth")

        # prediction on subset of train/val
        for dataset_name, hadm_ids in [("Train", train_hadm_ids), ("Validation", val_hadm_ids)]:
            random.shuffle(hadm_ids)
            hadm_to_process = hadm_ids[:min(num_hadms, len(hadm_ids))]
            for run in range(1, NUM_RUNS + 1):
                print(f"Run {run} predictions on {dataset_name} set for {category}...")
                for hadm_id in hadm_to_process:
                    seq = result[hadm_id][cat_key]
                    if len(seq) < 2:
                        continue
                    input_seq = seq[:-1]
                    ground_truth = seq[-1]
                    pred_item, pred_label, gt_label, is_correct = predict_next(
                        model, input_seq, ground_truth, item2idx, idx2item,
                        category=category,
                        itemid_label_mappings_flat=flat_itemid_label_mapping
                    )
                    all_prediction_rows.append({
                        "Run": run,
                        "Category": category,
                        "HADM_ID": hadm_id,
                        "Input_Sequence": ", ".join(map(str, input_seq)),
                        "Ground_Truth_ITEMID": ground_truth,
                        "Ground_Truth_Label": gt_label,
                        "Predicted_ITEMID": pred_item,
                        "Predicted_Label": pred_label,
                        "Is_Correct": is_correct,
                        "Dataset": dataset_name
                    })

    # save results to excel sheets
    pd.DataFrame(all_prediction_rows).to_excel("results/train_val_predictions.xlsx", index=False)
    pd.DataFrame(all_metrics).to_excel("results/train_val_metrics.xlsx", index=False)
    print("Saved prediction results and metrics.")

if __name__ == "__main__":
    main()
