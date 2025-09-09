# import os module for file path operations
import os
# import pandas for data manipulation
import pandas as pd
# import Counter for counting items
from collections import Counter
# import torch for tensor operations
import torch
import math
# import NROWS constant from info module
from info import NROWS, banned_items_file

# load mimic data
def load_mimic3_data(mimic_3data, nrows, banned_items=None):
    print(f"\n=== Loading MIMIC-III data from: {mimic_3data} ===")

    if banned_items is None:
        banned_items = set()

    # ensure all banned items are strings
    banned_items = set(str(x) for x in banned_items)

    # load admissions file
    admissions_file = os.path.join(mimic_3data, 'ADMISSIONS.csv')
    # check if admissions file exists
    if not os.path.exists(admissions_file):
        print(f" File not found: {admissions_file}")
        raise FileNotFoundError("ADMISSIONS.csv is required for admission times")

    # read admissions file
    admissions_df = pd.read_csv(
        # specify file path, specific columns
        admissions_file,
        usecols=['HADM_ID', 'ADMITTIME'],
        # set HADM_ID as nullable integer
        dtype={'HADM_ID': 'Int64'},
        # parse admittime as datetime
        parse_dates=['ADMITTIME']
    )

    # # create dict of admission times
    admission_times = dict(zip(admissions_df['HADM_ID'], admissions_df['ADMITTIME']))
    # print statement for debugging
    print(f" Loaded ADMISSIONS.csv ({len(admissions_df)} rows)")

    # define dictionary of data files and their properties
    data_files = {
        'chart_events': ('CHARTEVENTS.csv', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'input_events': ('INPUTEVENTS_MV.csv', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
        'lab_events': ('LABEVENTS.csv', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'microbiology_events': ('MICROBIOLOGYEVENTS.csv', ['HADM_ID', 'CHARTTIME'], 'SPEC_ITEMID'),
        'prescriptions': ('PRESCRIPTIONS.csv', ['HADM_ID', 'STARTDATE'], 'DRUG'),
        'procedure_events': ('PROCEDUREEVENTS_MV.csv', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
    }

    # dict to store data
    data_dicts = {}
    # stores chart event counts
    chart_event_counts = {}

    # process each category
    for key, (file_name, sort_cols, group_col) in data_files.items():
        file_path = os.path.join(mimic_3data, file_name)

        # debug: check file existence
        print(f"\n--- Checking category: '{key}' ---")
        if not os.path.exists(file_path):
            # Try .csv.gz alternative
            gz_path = file_path + ".gz"
            if os.path.exists(gz_path):
                file_path = gz_path
                compression_arg = 'gzip'
                print(f" Found compressed file: {gz_path}")
            else:
                print(f" File not found: {file_path} (.gz also missing) — skipping.")
                data_dicts[key] = {}
                continue
        else:
            compression_arg = None
            print(f" Found file: {file_path}")

        # Special case: chart_events
        if key == 'chart_events':
            print(f" Loading {file_name} for '{key}'...")
            df = pd.read_csv(
                file_path,
                compression=compression_arg,
                nrows=nrows,
                usecols=['HADM_ID', 'CHARTTIME', 'ITEMID'],
                dtype={'HADM_ID': 'Int64', 'ITEMID': 'Int64'},
                parse_dates=['CHARTTIME']
            )
            print(f"   → Loaded {len(df)} rows BEFORE filtering")

            df = df.dropna(subset=['HADM_ID', 'CHARTTIME', 'ITEMID'])
            df['ADMITTIME'] = df['HADM_ID'].map(admission_times)
            df = df.dropna(subset=['ADMITTIME'])
            df['TIME_DIFF'] = (df['CHARTTIME'] - df['ADMITTIME']).dt.total_seconds() / 3600

            # counts total events for the first 24 hours per admission
            before_filter = len(df)
            df = df[(df['TIME_DIFF'] >= 0) & (df['TIME_DIFF'] <= 24)]
            print(f"   → Filtered to first 24h: {len(df)} rows (was {before_filter})")

            # Count events
            chart_event_counts = df.groupby('HADM_ID').size().to_dict()
            for hadm_id in admission_times:
                chart_event_counts.setdefault(hadm_id, 0)

            # Store sequences
            sorted_df = df.sort_values(by=['HADM_ID', 'CHARTTIME'])
            data_dicts[key] = sorted_df.groupby('HADM_ID')['ITEMID'].apply(list).to_dict()
            print(f"Stored chart_item sequences for {len(data_dicts[key])} admissions")

        # All other categories
        else:
            print(f" Loading {file_name} for '{key}'...")
            dtype_for_cols = {}
            parse_dates_cols = []
            for col in sort_cols + [group_col]:
                if col in ('CHARTTIME', 'STARTTIME', 'STARTDATE'):
                    parse_dates_cols.append(col)
                elif col == 'HADM_ID':
                    dtype_for_cols[col] = 'Int64'
                elif col in ('ITEMID', 'SPEC_ITEMID'):
                    dtype_for_cols[col] = 'Int64'
                elif col == 'DRUG':
                    dtype_for_cols[col] = str
                else:
                    dtype_for_cols[col] = str

            df = pd.read_csv(
                file_path,
                compression=compression_arg,
                nrows=nrows,
                usecols=sort_cols + [group_col],
                dtype=dtype_for_cols if dtype_for_cols else None,
                parse_dates=parse_dates_cols if parse_dates_cols else None
            )
            print(f"   → Loaded {len(df)} rows BEFORE dropna")
            df = df.dropna(subset=[group_col])
            sorted_df = df.sort_values(by=sort_cols)
            data_dicts[key] = sorted_df.groupby('HADM_ID')[group_col].apply(list).to_dict()
            print(f"    Stored sequences for {len(data_dicts[key])} admissions")

    # Merge all dicts
    all_hadm_ids = set(admission_times.keys()).union(*[d.keys() for d in data_dicts.values()])
    merged_dict = {}
    for hadm_id in all_hadm_ids:
        current_dict = {
            'chart_items': data_dicts.get('chart_events', {}).get(hadm_id, []),
            'input_items': data_dicts.get('input_events', {}).get(hadm_id, []),
            'lab_items': data_dicts.get('lab_events', {}).get(hadm_id, []),
            'microbiology_items': data_dicts.get('microbiology_events', {}).get(hadm_id, []),
            'prescriptions_items': data_dicts.get('prescriptions', {}).get(hadm_id, []),
            'procedure_items': data_dicts.get('procedure_events', {}).get(hadm_id, []),
        }
        if any(current_dict.values()):
            merged_dict[hadm_id] = current_dict

    # 24/12 to get every 2 hours
    chart_event_avg_per_2hr = {hid: math.ceil(count / 12) for hid, count in chart_event_counts.items()}

    # Debug print top 5 rows
    chart_avg_df = pd.DataFrame({
        'HADM_ID': list(chart_event_avg_per_2hr.keys()),
        'Avg_Chart_Events_Per_2hr': list(chart_event_avg_per_2hr.values())
    })

    # check if it works
    print("Top 5 chart event avg per 2hr:")
    print(chart_avg_df.head(5))

    print(f"\n=== Finished loading all categories ===")
    print(f"Total admissions loaded: {len(merged_dict)}")
    print(f"Chart events count entries: {len(chart_event_avg_per_2hr)}\n")

    return merged_dict, chart_event_avg_per_2hr

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

    pad_idx = item2idx.get('<PAD>')
    if pad_idx is not None:
        weights[pad_idx] = 0.0  # PAD must have no influence

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

# save encoded sequences to excel
def save_sequences_to_excel(encoded_sequences, filepath='encoded_sequences.xlsx'):
    data = [{'HADM_ID': hadm_id,
             'Encoder_Input': ','.join(map(str, enc)),
             'Decoder_Input': ','.join(map(str, dec_in)),
             'Decoder_Output': ','.join(map(str, dec_out))}
            for hadm_id, enc, dec_in, dec_out in encoded_sequences]
    pd.DataFrame(data).to_excel(filepath, index=False)
    print(f"Sequences saved to {filepath}")

# save vocab to excel sheets
def save_vocab_to_excel(item2idx, class_weights=None, filepath='vocab.xlsx'):
    """
    Save the vocabulary and optionally the class weights to an Excel file.

    Args:
        item2idx (dict): Mapping from item string to index.
        class_weights (torch.Tensor, optional): Tensor of class weights.
        filepath (str): Excel file path to save to.
    """
    # prepare data for DataFrame: item, index, and optionally weight
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

# save chart event averages to Excel
def save_chart_event_averages(chart_event_avg_per_2hr, filepath='chart_event_averages.parquet'):
    df = pd.DataFrame({
        'HADM_ID': list(chart_event_avg_per_2hr.keys()),
        'Avg_Chart_Events_Per_2hr': list(chart_event_avg_per_2hr.values())
    })
    df.to_parquet(filepath, index=False)
    print(f"Chart event averages saved to {filepath}")

# main script
if __name__ == "__main__":
    mimic_data_dir = '/Users/amandali/Downloads/Mimic III'
    # load data and get chart event averages
    result, chart_event_avg_per_2hr = load_mimic3_data(mimic_data_dir, nrows=NROWS, banned_items=banned_items_file)

    # Save chart event averages
    save_chart_event_averages(chart_event_avg_per_2hr)

    # Save each category's sequences to Excel as a cache
    for cat_key in ['chart_items', 'input_items', 'lab_items',
                    'microbiology_items', 'prescriptions_items', 'procedure_items']:
        cache_path = os.path.join(mimic_data_dir, f"preprocessed_{cat_key}.parquet")

        data = []
        for hadm_id, cats in result.items():
            seq = cats.get(cat_key, [])
            data.append({
                'HADM_ID': hadm_id,
                'Category': cat_key,
                'Sequence': ','.join(map(str, seq)) if seq else ''
            })

        pd.DataFrame(data).to_parquet(cache_path, index=False)
        print(f" Saved {cat_key} cache to {cache_path} ({len(data)} rows)")

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

"""
# don't need this segment of code but will keep here in case needed in the future
def load_vocab_from_excel(filepath='vocab.xlsx'):
    df = pd.read_excel(filepath)
    item2idx = dict(zip(df['Item'], df['Index']))
    if 'Weight' in df.columns:
        weights_list = df['Weight'].fillna(1.0).tolist()  # fill missing with 1.0
        class_weights = torch.tensor(weights_list, dtype=torch.float)
    else:
        class_weights = None
    return item2idx, class_weights
"""
