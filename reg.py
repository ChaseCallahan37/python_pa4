import pandas as pd
import numpy as np
from statsmodels.formula.api import ols 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


HOUSE_FILE = "pa4_reg_data.txt"

def main():
    house_df = read_house_data()
    print(house_df[["sqft_living", "price", "grade", "floors", "age", "yr_built", "view"]].describe())
    ols_model = ols(formula="price ~ sqft_living + grade + floors + age + view", data=house_df).fit()
    
    house_df["price_pred"] = ols_model.predict(house_df[["sqft_living", "grade", "floors", "age", "view"]])

    print(house_df[["price", "price_pred"]])

    print(ols_model.summary())

    plt.scatter(x=house_df["sqft_living"], y=house_df["price"], s=1, c="Blue", alpha=.4)
    plt.scatter(x=house_df["sqft_living"], y=house_df["price_pred"], s=1, c="Orange", alpha=.4)

    sqft_mean = house_df["sqft_living"].mean()
    price_mean = house_df["price"].mean()

    plt.axline((sqft_mean, price_mean), (sqft_mean * 1.5, price_mean * 1.5))
    plt.show()


    mae = mean_absolute_error(house_df["price"], house_df["price_pred"])
    print(f"Mean Absolute Error: {mae}")

    mse = mean_squared_error(house_df["price"], house_df["price_pred"])
    rmse = mse ** .5
    print(f"Root Mean Squared Error: {rmse}")


def read_house_data() -> pd.DataFrame:
    house_df = pd.read_csv(HOUSE_FILE, delimiter="\t")
    house_df["age"] = house_df["yr_built"].apply(lambda x: 2015 - x if pd.notna(x) else np.nan)
    return house_df

main()