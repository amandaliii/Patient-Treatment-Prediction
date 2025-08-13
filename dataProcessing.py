# imports the os module for interacting with the operating system
import os
# imports pandas library as pd for data manipulation and analysis
import pandas as pd
# imports counter class for counting occurrences of items in sequences
from collections import Counter
# imports pytorch library for tensor operations
import torch
# imports nrows from info.py module
from info import NROWS
# imports math module for mathematical operations
import math

# defines function to load and process mimic-iii data
def load_mimic3_data(mimic_3data, nrows):
    # constructs file path for admissions.csv.gz
    admissions_file = os.path.join(mimic_3data, 'ADMISSIONS.csv.gz')
    # checks if admissions file exists
    if not os.path.exists(admissions_file):
        # prints error message if file is not found
        print(f"File not found: {admissions_file}")
        # raises exception if file is missing
        raise FileNotFoundError("ADMISSIONS.csv.gz is required for admission times")

    # reads admissions.csv.gz into a dataframe
    admissions_df = pd.read_csv(
        # specifies the file path
        admissions_file,
        # indicates gzip compression
        compression='gzip',
        # loads only hadm_id and admittime columns
        usecols=['HADM_ID', 'ADMITTIME'],
        # sets hadm_id as nullable integer type
        dtype={'HADM_ID': 'Int64'},
        # parses admittime as datetime
        parse_dates=['ADMITTIME']
    )
    # creates dictionary mapping hadm_id to admittime
    admission_times = dict(zip(admissions_df['HADM_ID'], admissions_df['ADMITTIME']))

    # defines dictionary mapping data categories to file names sort columns and grouping column
    data_files = {
        # chart events configuration
        'chart_events': ('CHARTEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        # input events configuration
        'input_events': ('INPUTEVENTS_MV.csv.gz', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
        # lab events configuration
        'lab_events': ('LABEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        # microbiology events configuration
        'microbiology_events': ('MICROBIOLOGYEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'SPEC_ITEMID'),
        # prescriptions configuration
        'prescriptions': ('PRESCRIPTIONS.csv.gz', ['HADM_ID', 'STARTDATE'], 'DRUG'),
        # procedure events configuration
        'procedure_events': ('PROCEDUREEVENTS_MV.csv.gz', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
    }

    # initializes empty dictionary to store grouped data
    data_dicts = {}
    # initializes empty dictionary to store chart event counts
    chart_event_counts = {}

    # iterates over each category in data_files
    for key, (file_name, sort_cols, group_col) in data_files.items():
        # constructs full file path for current category
        file_path = os.path.join(mimic_3data, file_name)
        # checks if file exists
        if not os.path.exists(file_path):
            # assigns empty dictionary for missing file
            data_dicts[key] = {}
            # skips to next category
            continue

        # special handling for chart_events
        if key == 'chart_events':
            # reads chart events csv into dataframe
            df = pd.read_csv(
                # specifies file path
                file_path,
                # indicates gzip compression
                compression='gzip',
                # limits number of rows read
                nrows=nrows,
                # loads specified columns
                usecols=['HADM_ID', 'CHARTTIME', 'ITEMID'],
                # sets nullable integer types
                dtype={'HADM_ID': 'Int64', 'ITEMID': 'Int64'},
                # parses charttime as datetime
                parse_dates=['CHARTTIME']
            )
            # removes rows with missing values in specified columns
            df = df.dropna(subset=['HADM_ID', 'CHARTTIME', 'ITEMID'])
            # adds admittime column by mapping hadm_id
            df['ADMITTIME'] = df['HADM_ID'].map(admission_times)
            # removes rows with missing admittime
            df = df.dropna(subset=['ADMITTIME'])
            # calculates time difference in hours
            df['TIME_DIFF'] = (df['CHARTTIME'] - df['ADMITTIME']).dt.total_seconds() / 3600
            # filters events within first 24 hours
            df = df[(df['TIME_DIFF'] >= 0) & (df['TIME_DIFF'] <= 24)]
            # counts events per hadm_id
            chart_event_counts = df.groupby('HADM_ID').size().to_dict()
            # sets count to 0 for hadm_ids with no chart events
            for hadm_id in admission_times:
                # ensures all hadm_ids have a count
                chart_event_counts.setdefault(hadm_id, 0)
            # sorts dataframe by hadm_id and charttime
            sorted_df = df.sort_values(by=['HADM_ID', 'CHARTTIME'])
            # groups by hadm_id and converts itemid to lists
            data_dicts[key] = sorted_df.groupby('HADM_ID')['ITEMID'].apply(list).to_dict()
            # prints number of rows loaded
            print(f"Loaded {file_name} with {len(df)} rows after filtering")
        # handling for other categories
        else:
            # initializes dictionary for column data types
            dtype_for_cols = {}
            # initializes list for date columns
            parse_dates_cols = []
            # iterates over sort and grouping columns
            for col in sort_cols + [group_col]:
                # checks if column is a date/time column
                if col in ('CHARTTIME', 'STARTTIME', 'STARTDATE'):
                    # adds column to date parsing list
                    parse_dates_cols.append(col)
                # checks if column is hadm_id
                elif col == 'HADM_ID':
                    # sets hadm_id as nullable integer
                    dtype_for_cols[col] = 'Int64'
                # checks if column is itemid or spec_itemid
                elif col in ('ITEMID', 'SPEC_ITEMID'):
                    # sets as nullable integer
                    dtype_for_cols[col] = 'Int64'
                # checks if column is drug
                elif col == 'DRUG':
                    # sets drug as string type
                    dtype_for_cols[col] = str
                # default case for other columns
                else:
                    # sets as string type
                    dtype_for_cols[col] = str
            # reads csv file for current category
            df = pd.read_csv(
                # specifies file path
                file_path,
                # indicates gzip compression
                compression='gzip',
                # limits number of rows read
                nrows=nrows,
                # loads sort and grouping columns
                usecols=sort_cols + [group_col],
                # applies specified data types
                dtype=dtype_for_cols if dtype_for_cols else None,
                # parses specified date columns
                parse_dates=parse_dates_cols if parse_dates_cols else None
            )
            # removes rows with missing grouping column
            df = df.dropna(subset=[group_col])
            # sorts dataframe by specified columns
            sorted_df = df.sort_values(by=sort_cols)
            # groups by hadm_id and converts grouping column to lists
            data_dicts[key] = sorted_df.groupby('HADM_ID')[group_col].apply(list).to_dict()
            # prints number of rows loaded
            print(f"Loaded {file_name} with {len(df)} rows")

    # creates set of all unique hadm_ids
    all_hadm_ids = set(admission_times.keys()).union(*[d.keys() for d in data_dicts.values()])
    # initializes empty dictionary for merged data
    merged_dict = {}
    # iterates over all unique hadm_ids
    for hadm_id in all_hadm_ids:
        # creates dictionary for current hadm_id
        current_dict = {
            # gets chart_events sequence or empty list
            'chart_items': data_dicts.get('chart_events', {}).get(hadm_id, []),
            # gets input_events sequence or empty list
            'input_items': data_dicts.get('input_events', {}).get(hadm_id, []),
            # gets lab_events sequence or empty list
            'lab_items': data_dicts.get('lab_events', {}).get(hadm_id, []),
            # gets microbiology_events sequence or empty list
            'microbiology_items': data_dicts.get('microbiology_events', {}).get(hadm_id, []),
            # gets prescriptions sequence or empty list
            'prescriptions_items': data_dicts.get('prescriptions', {}).get(hadm_id, []),
            # gets procedure_events sequence or empty list
            'procedure_items': data_dicts.get('procedure_events', {}).get(hadm_id, []),
        }
        # checks if any category has non-empty sequence
        if any(current_dict.values()):
            # adds current_dict to merged_dict
            merged_dict[hadm_id] = current_dict

    # creates dictionary for average chart events per 2-hour period
    chart_event_avg_per_2hr = {
        # calculates average by dividing count by 12 and flooring
        hid: math.floor(count / 12)
        # iterates over chart event counts
        for hid, count in chart_event_counts.items()
    }
    # returns merged dictionary and chart event averages
    return merged_dict, chart_event_avg_per_2hr

# defines function to build vocabulary and class weights
def build_vocab_with_weights(sequences):
    # initializes counter object for item occurrences
    item_counts = Counter()
    # iterates over each sequence
    for seq in sequences:
        # updates counts by converting items to strings
        item_counts.update(str(item) for item in seq)

    # defines special tokens for padding unknown begin and end
    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    # creates vocabulary list with special tokens and sorted items
    vocab_items = special_tokens + [item for item, _ in item_counts.most_common()]

    # creates dictionary mapping items to indices
    item2idx = {item: idx for idx, item in enumerate(vocab_items)}
    # creates reverse dictionary mapping indices to items
    idx2item = {idx: item for item, idx in item2idx.items()}

    # calculates total number of item occurrences
    total_count = sum(item_counts.values())
    # initializes list for class weights
    weights = []
    # iterates over vocabulary items
    for item in vocab_items:
        # checks if item is a special token
        if item in special_tokens:
            # assigns weight of 1.0 to special tokens
            weights.append(1.0)
        # for regular items
        else:
            # gets count of the item
            count = item_counts[item]
            # calculates inverse frequency weight
            weights.append(total_count / (len(item_counts) * count))

    # converts weights list to pytorch tensor
    class_weights = torch.tensor(weights, dtype=torch.float)
    # returns item-to-index index-to-item and class weights
    return item2idx, idx2item, class_weights

# defines function to create encoder-decoder sequences
def build_encoder_decoder_sequences(merged_dict, vocab):
    # initializes list to store encoded sequences
    encoded_sequences = []
    # iterates over each hadm_id and data
    for hadm_id, data in merged_dict.items():
        # initializes list for tokens
        token_list = []
        # iterates over categories
        for cat in ['chart_items', 'input_items', 'lab_items',
                    'microbiology_items', 'prescriptions_items', 'procedure_items']:
            # converts category items to strings and adds to token_list
            token_list.extend(str(code) for code in data[cat])
        # skips sequences with fewer than 3 tokens
        if len(token_list) < 3:
            # continues to next hadm_id
            continue
        # maps tokens to vocabulary indices
        token_ids = [vocab.get(code, vocab['<UNK>']) for code in token_list]
        # creates encoder input by excluding last token
        encoder_input = token_ids[:-1]
        # creates decoder target by excluding first token
        decoder_target = token_ids[1:]
        # creates decoder input by adding <bos> token
        decoder_input = [vocab['<BOS>']] + decoder_target
        # creates decoder output by adding <eos> token
        decoder_output = decoder_target + [vocab['<EOS>']]
        # adds tuple of hadm_id and sequences to list
        encoded_sequences.append((hadm_id, encoder_input, decoder_input, decoder_output))
    # returns list of encoded sequences
    return encoded_sequences

# defines function to save encoded sequences to excel
def save_sequences_to_excel(encoded_sequences, filepath='encoded_sequences.xlsx'):
    # creates list of dictionaries for each sequence
    data = [{'HADM_ID': hadm_id,
             # joins encoder input indices with commas
             'Encoder_Input': ','.join(map(str, enc)),
             # joins decoder input indices with commas
             'Decoder_Input': ','.join(map(str, dec_in)),
             # joins decoder output indices with commas
             'Decoder_Output': ','.join(map(str, dec_out))}
            # iterates over encoded sequences
            for hadm_id, enc, dec_in, dec_out in encoded_sequences]
    # converts data to dataframe and saves to excel
    pd.DataFrame(data).to_excel(filepath, index=False)
    # prints confirmation of saving
    print(f"Sequences saved to {filepath}")

# defines function to save vocabulary and weights to excel
def save_vocab_to_excel(item2idx, class_weights=None, filepath='vocab.xlsx'):
    # docstring explaining function purpose and arguments
    """
    Save the vocabulary and optionally the class weights to an Excel file.

    Args:
        item2idx (dict): Mapping from item string to index.
        class_weights (torch.Tensor, optional): Tensor of class weights.
        filepath (str): Excel file path to save to.
    """
    # initializes list for vocabulary data
    data = []
    # iterates over item-to-index mappings
    for item, idx in item2idx.items():
        # initializes weight as none
        weight = None
        # checks if weights exist and index is valid
        if class_weights is not None and idx < len(class_weights):
            # converts tensor weight to python float
            weight = class_weights[idx].item()
        # adds item index and weight to data list
        data.append({'Item': item, 'Index': idx, 'Weight': weight})

    # creates dataframe from data list
    df = pd.DataFrame(data)

    # saves dataframe to excel without index
    df.to_excel(filepath, index=False)
    # prints confirmation of saving
    print(f"Vocabulary saved to Excel file: {filepath}")

# defines function to load vocabulary and weights from excel
def load_vocab_from_excel(filepath='vocab.xlsx'):
    # load vocabulary and class weights from an Excel file saved by save_vocab_to_excel.
    # Returns: item2idx mapping from item string to index, and class weights
    # reads excel file into dataframe
    df = pd.read_excel(filepath)
    # creates item-to-index dictionary
    item2idx = dict(zip(df['Item'], df['Index']))
    # checks if weight column exists
    if 'Weight' in df.columns:
        # fills missing weights with 1.0 and converts to list
        weights_list = df['Weight'].fillna(1.0).tolist()
        # converts weights to pytorch tensor
        class_weights = torch.tensor(weights_list, dtype=torch.float)
    # if no weight column
    else:
        # sets weights to none
        class_weights = None
    # returns item-to-index and class weights
    return item2idx, class_weights

# defines function to save chart event averages to excel
def save_chart_event_averages(chart_event_avg_per_2hr, filepath='chart_event_averages.xlsx'):
    # creates dataframe from chart event averages
    df = pd.DataFrame({
        # column for hadm_ids
        'HADM_ID': list(chart_event_avg_per_2hr.keys()),
        # column for average event counts
        'Avg_Chart_Events_Per_2hr': list(chart_event_avg_per_2hr.values())
    })
    # saves dataframe to excel without index
    df.to_excel(filepath, index=False)
    # prints confirmation of saving
    print(f"Chart event averages saved to {filepath}")

# checks if script is run directly
if __name__ == "__main__":
    # sets directory path for mimic-iii data
    mimic_data_dir = '/Users/amandali/Downloads/Mimic III'
    # calls load_mimic3_data to load data and compute averages
    result, chart_event_avg_per_2hr = load_mimic3_data(mimic_data_dir, nrows=NROWS)

    # saves chart event averages to excel
    save_chart_event_averages(chart_event_avg_per_2hr)

    # initializes list for all sequences
    all_sequences = []
    # iterates over data dictionaries in result
    for data in result.values():
        # initializes temporary sequence list
        seq = []
        # iterates over categories
        for cat in ['chart_items', 'input_items', 'lab_items',
                    'microbiology_items', 'prescriptions_items', 'procedure_items']:
            # converts items to strings and adds to sequence
            seq.extend(str(item) for item in data[cat])
        # checks if sequence is non-empty
        if seq:
            # adds sequence to all_sequences
            all_sequences.append(seq)

    # builds vocabulary and weights from sequences
    item2idx, idx2item, class_weights = build_vocab_with_weights(all_sequences)

    # saves vocabulary and weights to excel
    save_vocab_to_excel(item2idx, class_weights, filepath='mimic_vocab.xlsx')

    # creates encoder-decoder sequences
    encoded_sequences = build_encoder_decoder_sequences(result, item2idx)

    # saves encoded sequences to excel
    save_sequences_to_excel(encoded_sequences, filepath='encoded_sequences.xlsx')

    # prints completion message
    print("Done!")
