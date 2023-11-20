import pandas as pd
import numpy as np

HOUSE_FILE = "pa4_reg_data.txt"

def main():
    house_df = read_house_data()
    print(house_df[["sqft_living", "price", "grade", "floors", "age", "yr_built", "view"]].describe())

def read_house_data() -> pd.DataFrame:
    house_df = pd.read_csv(HOUSE_FILE, delimiter="\t")
    house_df["age"] = house_df["yr_built"].apply(lambda x: 2015 - x if pd.notna(x) else np.nan)
    return house_df

main()