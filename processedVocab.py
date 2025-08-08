import os
import pandas as pd
from collections import Counter
import torch

# load mimic data
def load_mimic3_data(mimic_3data, nrows):
    # chart events: observations and measurements
    # input events: administration of fluids, medications, or other substances
    # data dictionary mapping - file name, sort columns, and grouping column
    data_files = {
        'chart_events': ('CHARTEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'input_events': ('INPUTEVENTS_MV.csv.gz', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
        'lab_events': ('LABEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'microbiology_events': ('MICROBIOLOGYEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'SPEC_ITEMID'),
        'prescriptions': ('PRESCRIPTIONS.csv.gz', ['HADM_ID', 'STARTDATE'], 'DRUG'),
        'procedure_events': ('PROCEDUREEVENTS_MV.csv.gz', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
    }

    # store grouped data for each category
    data_dicts = {}
    for key, (file_name, sort_cols, group_col) in data_files.items():
        file_path = os.path.join(mimic_3data, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            data_dicts[key] = {}
            continue
        try:
            dtype_mapping = {'HADM_ID': 'Int64', 'ITEMID': 'Int64', 'SPEC_ITEMID': 'Int64', 'DRUG': str}
            dtype_for_cols = {col: dtype_mapping.get(col, str) for col in sort_cols}

            # if group_col is DRUG (string), keep as str
            if group_col == 'DRUG':
                dtype_for_cols[group_col] = str
                df = pd.read_csv(file_path, compression='gzip', nrows=nrows,
                                 usecols=sort_cols + [group_col], dtype=dtype_for_cols)
            else:
                dtype_for_cols[group_col] = 'Int64'
                df = pd.read_csv(file_path, compression='gzip', nrows=nrows,
                                 usecols=sort_cols + [group_col], dtype=dtype_for_cols)
                # drop missing values:
                df = df.dropna(subset=[group_col])
                # cast to int using pd.Int64Dtype
                df[group_col] = df[group_col].astype('Int64')

            if df.empty:
                data_dicts[key] = {}
            else:
                sorted_df = df.sort_values(by=sort_cols)
                data_dicts[key] = sorted_df.groupby('HADM_ID')[group_col].apply(list).to_dict()
                print(f"Loaded {file_name} with {len(df)} rows")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            data_dicts[key] = {}

    all_hadm_ids = set().union(*[d.keys() for d in data_dicts.values()])
    merged_dict = {}
    for hadm_id in all_hadm_ids:
        current_dict = {
            'chart_items': data_dicts['chart_events'].get(hadm_id, []),
            'input_items': data_dicts['input_events'].get(hadm_id, []),
            'lab_items': data_dicts['lab_events'].get(hadm_id, []),
            'microbiology_items': data_dicts['microbiology_events'].get(hadm_id, []),
            'prescriptions_items': data_dicts['prescriptions'].get(hadm_id, []),
            'procedure_items': data_dicts['procedure_events'].get(hadm_id, []),
        }
        if any(len(items) > 0 for items in current_dict.values()):
            merged_dict[hadm_id] = current_dict

    return merged_dict

# build vocab and weights
def build_vocab_with_weights(sequences):
    item_counts = Counter()
    for seq in sequences:
        item_counts.update(str(item) for item in seq)

    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    vocab_items = special_tokens + [item for item, _ in item_counts.most_common()]

    item2idx = {item: idx for idx, item in enumerate(vocab_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}

    total_count = sum(item_counts.values())
    weights = []
    for item in vocab_items:
        if item in special_tokens:
            weights.append(1.0)
        else:
            count = item_counts[item]
            weights.append(total_count / (len(item_counts) * count))

    class_weights = torch.tensor(weights, dtype=torch.float)
    return item2idx, idx2item, class_weights

# build encoder-decoder sequences
def build_encoder_decoder_sequences(merged_dict, vocab):
    encoded_sequences = []
    for hadm_id, data in merged_dict.items():
        token_list = []
        for cat in ['chart_items', 'input_items', 'lab_items',
                    'microbiology_items', 'prescriptions_items', 'procedure_items']:
            token_list.extend(str(code) for code in data[cat])
        if len(token_list) < 3:
            continue
        token_ids = [vocab.get(code, vocab['<UNK>']) for code in token_list]
        encoder_input = token_ids[:-1]
        decoder_target = token_ids[1:]
        decoder_input = [vocab['<BOS>']] + decoder_target
        decoder_output = decoder_target + [vocab['<EOS>']]
        encoded_sequences.append((hadm_id, encoder_input, decoder_input, decoder_output))
    return encoded_sequences

# save vocab to excel sheets
def save_vocab_to_excel(item2idx, class_weights, filepath='vocab.xlsx'):
    data = [{'Item': item, 'Index': idx, 'Weight': class_weights[idx].item()}
            for item, idx in item2idx.items()]
    pd.DataFrame(data).to_excel(filepath, index=False)
    print(f"Vocabulary saved to {filepath}")

def save_sequences_to_excel(encoded_sequences, filepath='encoded_sequences.xlsx'):
    data = [{'HADM_ID': hadm_id,
             'Encoder_Input': ','.join(map(str, enc)),
             'Decoder_Input': ','.join(map(str, dec_in)),
             'Decoder_Output': ','.join(map(str, dec_out))}
            for hadm_id, enc, dec_in, dec_out in encoded_sequences]
    pd.DataFrame(data).to_excel(filepath, index=False)
    print(f"Sequences saved to {filepath}")

# main script
if __name__ == "__main__":
    mimic_data_dir = '/Users/amandali/Downloads/Mimic III'
    result = load_mimic3_data(mimic_data_dir, nrows=2000000)

    # extract sequences for vocab building
    all_sequences = []
    for data in result.values():
        seq = []
        for cat in ['chart_items', 'input_items', 'lab_items',
                    'microbiology_items', 'prescriptions_items', 'procedure_items']:
            seq.extend(str(item) for item in data[cat])
        if seq:
            all_sequences.append(seq)

    # build vocab with weights
    item2idx, idx2item, class_weights = build_vocab_with_weights(all_sequences)

    # save vocab
    save_vocab_to_excel(item2idx, class_weights, filepath='mimic_vocab.xlsx')

    # build encoder–decoder sequences
    encoded_sequences = build_encoder_decoder_sequences(result, item2idx)

    # save sequences
    save_sequences_to_excel(encoded_sequences, filepath='encoded_sequences.xlsx')

    print("Done!")
