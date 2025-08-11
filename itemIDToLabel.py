import csv
import pandas as pd

def create_itemid_label_mapping(csv_file_path):
    itemid_label_map = {}
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                itemid = row['ITEMID']
                label = row['LABEL']
                itemid_label_map[itemid] = label
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return itemid_label_map

# get labels for lab item IDs
def load_labitems_labels(csv_file_path):
    # empt dict to store lab item labels
    labitem_labels = {}
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                itemid = row['ITEMID']
                label = row['LABEL']
                # store label keyed by ITEMID in dictionary
                labitem_labels[itemid] = label
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return labitem_labels


def main():
    # path for the two files
    d_items_path = '/Users/amandali/Downloads/Mimic III/D_ITEMS.csv'
    d_labitems_path = '/Users/amandali/Downloads/Mimic III/D_LABITEMS.csv'

    # load itemID-label mapping as a flat dict
    itemid_label_map = create_itemid_label_mapping(d_items_path)

    # load lab item labels
    lab_items_labels = load_labitems_labels(d_labitems_path)

    # merge lab items labels into main mapping (overwriting or adding)
    itemid_label_map.update(lab_items_labels)

    # convert to DataFrame for export or further use
    mapping_rows = [{'ItemID': k, 'Label': v} for k, v in itemid_label_map.items()]
    df_mappings = pd.DataFrame(mapping_rows)

    output_file = 'idlabel_mappings.xlsx'
    df_mappings.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Data successfully saved to '{output_file}'.")

if __name__ == "__main__":
    main()
