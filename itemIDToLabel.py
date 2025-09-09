# import pandas for data manipulation
import pandas as pd

# function to load ITEMID to LABEL mappings
def load_itemid_label_mappings_parquet(d_items_path, d_labitems_path, cache_file="idlabel_map.parquet"):
    print("Loading D_ITEMS.csv...")
    # read items csv with specific columns and data types
    df_items = pd.read_csv(d_items_path, usecols=["ITEMID", "LABEL"], dtype={"ITEMID": "Int64", "LABEL": str})
    print("\nFirst 5 rows of D_ITEMS.csv:")
    print(df_items.head(5))

    print("Loading D_LABITEMS.csv...")
    # repeat with lab items csv
    df_labitems = pd.read_csv(d_labitems_path, usecols=["ITEMID", "LABEL"], dtype={"ITEMID": "Int64", "LABEL": str})

    # combine items and lab items dataframes
    combined_df = pd.concat([df_items, df_labitems], ignore_index=True)
    # remove duplicates, keeping last entry
    combined_df = combined_df.drop_duplicates(subset=["ITEMID"], keep="last")
    # save combined dataframe to parquet cache
    combined_df.to_parquet(cache_file, index=False)
    print(f"Saved mapping to Parquet cache: {cache_file}")
    return dict(zip(combined_df["ITEMID"], combined_df["LABEL"]))

def main():
    # path for items and labitems
    d_items_path = "/Users/amandali/Downloads/Mimic III/D_ITEMS.csv"
    d_labitems_path = "/Users/amandali/Downloads/Mimic III/D_LABITEMS_NEW.csv"
    # load itemid to label mappings
    mapping = load_itemid_label_mappings_parquet(d_items_path, d_labitems_path)
    print(f"First 5 mappings:\n{list(mapping.items())[:5]}")
    # debug specific ITEMID
    test_itemid = 220179
    print(f"ITEMID {test_itemid} mapped to: {mapping.get(test_itemid, 'Not found')}")

if __name__ == "__main__":
    main()

"""
# don't need this segment of code but will keep here in case needed in the future
# defines function to load itemid to label mapping from excel
def load_itemid_label_map_from_excel(excel_path):
    # reads excel file into a dataframe
    df = pd.read_excel(excel_path)

    # checks if itemid and label columns exist in dataframe
    if 'ITEMID' not in df.columns or 'LABEL' not in df.columns:
        # raises error if required columns are missing
        raise ValueError("Excel file must contain 'ITEMID' and 'LABEL' columns")

    # creates and returns dictionary mapping itemid to label
    return dict(zip(df['ITEMID'], df['LABEL']))
"""
