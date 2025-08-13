import os
import pandas as pd
from collections import Counter
import torch
from info import NROWS

# load mimic data
def load_mimic3_data(mimic_3data, nrows):
    # Load ADMISSIONS.csv.gz to get admission times
    admissions_file = os.path.join(mimic_3data, 'ADMISSIONS.csv')
    if not os.path.exists(admissions_file):
        print(f"File not found: {admissions_file}")
        raise FileNotFoundError("ADMISSIONS.csv.gz is required for admission times")

    # Load admissions data with relevant columns
    admissions_df = pd.read_csv(
        admissions_file,
        usecols=['HADM_ID', 'ADMITTIME'],
        dtype={'HADM_ID': 'Int64'},
        parse_dates=['ADMITTIME']
    )
    # Create a dictionary mapping HADM_ID to ADMITTIME
    admission_times = dict(zip(admissions_df['HADM_ID'], admissions_df['ADMITTIME']))

    # chart events: observations and measurements
    # input events: administration of fluids, medications, or other substances
    # data dictionary mapping - file name, sort columns, and grouping column
    data_files = {
        'chart_events': ('CHARTEVENTS.csv', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'input_events': ('INPUTEVENTS_MV.csv', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
        'lab_events': ('LABEVENTS.csv', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'microbiology_events': ('MICROBIOLOGYEVENTS.csv', ['HADM_ID', 'CHARTTIME'], 'SPEC_ITEMID'),
        'prescriptions': ('PRESCRIPTIONS.csv', ['HADM_ID', 'STARTDATE'], 'DRUG'),
        'procedure_events': ('PROCEDUREEVENTS_MV.csv', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
    }

    # store grouped data for each category
    data_dicts = {}
    # to store total chart events in first 24 hours per HADM_ID
    chart_event_counts = {}

    for key, (file_name, sort_cols, group_col) in data_files.items():
        file_path = os.path.join(mimic_3data, file_name)
        if not os.path.exists(file_path):
            data_dicts[key] = {}
            continue

        if key == 'chart_events':
            df = pd.read_csv(
                file_path,
                nrows=nrows,
                usecols=['HADM_ID', 'CHARTTIME', 'ITEMID'],
                dtype={'HADM_ID': 'Int64', 'ITEMID': 'Int64'},
                parse_dates=['CHARTTIME']
            )
            df = df.dropna(subset=['HADM_ID', 'CHARTTIME', 'ITEMID'])
            df['ADMITTIME'] = df['HADM_ID'].map(admission_times)
            df = df.dropna(subset=['ADMITTIME'])
            df['TIME_DIFF'] = (df['CHARTTIME'] - df['ADMITTIME']).dt.total_seconds() / 3600
            df = df[(df['TIME_DIFF'] >= 0) & (df['TIME_DIFF'] <= 24)]
            chart_event_counts = df.groupby('HADM_ID').size().to_dict()
            for hadm_id in admission_times:
                chart_event_counts.setdefault(hadm_id, 0)
            sorted_df = df.sort_values(by=['HADM_ID', 'CHARTTIME'])
            data_dicts[key] = sorted_df.groupby('HADM_ID')['ITEMID'].apply(list).to_dict()
            print(f"Loaded {file_name} with {len(df)} rows after filtering")
        else:
            # Detect and set the right types
            dtype_for_cols = {}
            parse_dates_cols = []
            for col in sort_cols + [group_col]:
                if col in ('CHARTTIME', 'STARTTIME', 'STARTDATE'):
                    parse_dates_cols.append(col)  # let pandas parse dates
                elif col == 'HADM_ID':
                    dtype_for_cols[col] = 'Int64'
                elif col in ('ITEMID', 'SPEC_ITEMID'):
                    dtype_for_cols[col] = 'Int64'
                elif col == 'DRUG':
                    dtype_for_cols[col] = str
                else:
                    dtype_for_cols[col] = str  # default
            df = pd.read_csv(
                file_path,
                nrows=nrows,
                usecols=sort_cols + [group_col],
                dtype=dtype_for_cols if dtype_for_cols else None,
                parse_dates=parse_dates_cols if parse_dates_cols else None
            )
            df = df.dropna(subset=[group_col])
            sorted_df = df.sort_values(by=sort_cols)
            data_dicts[key] = sorted_df.groupby('HADM_ID')[group_col].apply(list).to_dict()
            print(f"Loaded {file_name} with {len(df)} rows")

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

    chart_event_avg_per_2hr = {hid: count / 12 for hid, count in chart_event_counts.items()}
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

# save chart event averages to Excel
def save_chart_event_averages(chart_event_avg_per_2hr, filepath='chart_event_averages.xlsx'):
    df = pd.DataFrame({
        'HADM_ID': list(chart_event_avg_per_2hr.keys()),
        'Avg_Chart_Events_Per_2hr': list(chart_event_avg_per_2hr.values())
    })
    df.to_excel(filepath, index=False)
    print(f"Chart event averages saved to {filepath}")

# main script
if __name__ == "__main__":
    mimic_data_dir = '/Users/amandali/Downloads/Mimic III'
    # load data and get chart event averages
    result, chart_event_avg_per_2hr = load_mimic3_data(mimic_data_dir, nrows=NROWS)

    # Save chart event averages
    save_chart_event_averages(chart_event_avg_per_2hr)

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
