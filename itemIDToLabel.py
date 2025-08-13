import pandas as pd
import os

def load_itemid_label_mappings_excel(d_items_path, d_labitems_path, cache_file="idlabel_map.xlsx"):
    # load ITEMID -> LABEL mapping from D_ITEMS.csv and D_LABITEMS.csv.
    # if cache file exists, load it
    if os.path.exists(cache_file):
        df_cache = pd.read_excel(cache_file)
        mapping = dict(zip(df_cache["ITEMID"], df_cache["LABEL"]))
        print(f"Loaded mapping from Excel cache: {cache_file} ({len(mapping)} entries)")
        return mapping

    # otherwise, read CSVs directly
    print("Loading D_ITEMS.csv...")
    df_items = pd.read_csv(d_items_path, usecols=["ITEMID", "LABEL"])
    print("Loading D_LABITEMS.csv...")
    df_labitems = pd.read_csv(d_labitems_path, usecols=["ITEMID", "LABEL"])

    # merge — D_LABITEMS overwrites D_ITEMS where ITEMID matches
    combined_df = pd.concat([df_items, df_labitems], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["ITEMID"], keep="last")

    # save to Excel for caching
    combined_df.to_excel(cache_file, index=False, engine="openpyxl")
    print(f"Saved mapping to Excel cache: {cache_file}")

    # return as dict
    return dict(zip(combined_df["ITEMID"], combined_df["LABEL"]))

# get labels for lab item IDs
def load_itemid_label_map_from_excel(excel_path):
    # loads an ITEMID -> LABEL mapping from an Excel file
    df = pd.read_excel(excel_path)

    # ensure columns are named correctly
    if 'ITEMID' not in df.columns or 'LABEL' not in df.columns:
        raise ValueError("Excel file must contain 'ITEMID' and 'LABEL' columns")

    # convert DataFrame to dict {ITEMID: LABEL}
    return dict(zip(df['ITEMID'], df['LABEL']))

def main():
    d_items_path = "/Users/amandali/Downloads/Mimic III/D_ITEMS.csv"
    d_labitems_path = "/Users/amandali/Downloads/Mimic III/D_LABITEMS.csv"

    mapping = load_itemid_label_mappings_excel(d_items_path, d_labitems_path)

    # testing first five
    print(f"First 5 mappings:\n{list(mapping.items())[:5]}")

if __name__ == "__main__":
    main()
