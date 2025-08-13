# imports pandas library for data manipulation and analysis
import pandas as pd
# imports os module for interacting with the operating system
import os

# defines function to load itemid to label mappings from csv or excel cache
def load_itemid_label_mappings_excel(d_items_path, d_labitems_path, cache_file="idlabel_map.xlsx"):
    # checks if cache excel file exists
    if os.path.exists(cache_file):
        # reads cache file into a dataframe
        df_cache = pd.read_excel(cache_file)
        # creates dictionary mapping itemid to label from cache
        mapping = dict(zip(df_cache["ITEMID"], df_cache["LABEL"]))
        # prints confirmation of loading cache with entry count
        print(f"Loaded mapping from Excel cache: {cache_file} ({len(mapping)} entries)")
        # returns cached mapping dictionary
        return mapping

    # prints message indicating d_items.csv is being loaded
    print("Loading D_ITEMS.csv...")
    # reads d_items.csv into a dataframe with itemid and label columns
    df_items = pd.read_csv(d_items_path, usecols=["ITEMID", "LABEL"])
    # prints message indicating d_labitems.csv is being loaded
    print("Loading D_LABITEMS.csv...")
    # reads d_labitems.csv into a dataframe with itemid and label columns
    df_labitems = pd.read_csv(d_labitems_path, usecols=["ITEMID", "LABEL"])

    # concatenates d_items and d_labitems dataframes
    combined_df = pd.concat([df_items, df_labitems], ignore_index=True)
    # removes duplicate itemids keeping the last entry
    combined_df = combined_df.drop_duplicates(subset=["ITEMID"], keep="last")

    # saves combined dataframe to excel for caching
    combined_df.to_excel(cache_file, index=False, engine="openpyxl")
    # prints confirmation of saving to cache
    print(f"Saved mapping to Excel cache: {cache_file}")

    # creates and returns dictionary mapping itemid to label
    return dict(zip(combined_df["ITEMID"], combined_df["LABEL"]))

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

# defines main function for script execution
def main():
    # sets file path for d_items.csv
    d_items_path = "/Users/amandali/Downloads/Mimic III/D_ITEMS.csv"
    # sets file path for d_labitems.csv
    d_labitems_path = "/Users/amandali/Downloads/Mimic III/D_LABITEMS.csv"

    # loads itemid to label mappings using function
    mapping = load_itemid_label_mappings_excel(d_items_path, d_labitems_path)

    # prints first five mappings for testing
    print(f"First 5 mappings:\n{list(mapping.items())[:5]}")

if __name__ == "__main__":
    main()
