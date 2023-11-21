# Extras:
# 1.    Added menu for choosing which analysis you would like to see
#       as well as handling bad input

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


HOUSE_FILE = "pa4_reg_data.txt"

def main():
    house_df = read_house_data()
    ols_model = ols(formula="price ~ sqft_living + grade + floors + age + view", data=house_df).fit()
    house_df["price_pred"] = ols_model.predict(house_df[["sqft_living", "grade", "floors", "age", "view"]])

    user_choice = main_menu_choice()
    while user_choice != "7":
        if(user_choice == "1"):
            general_summary(house_df)
        elif(user_choice == "2"):
            show_price_prediction(house_df)
        elif(user_choice == "3"):
            show_ols_summary(ols_model)
        elif(user_choice == "4"):
            chart_pred_price(house_df)
        elif(user_choice == "5"):
            show_mae(house_df)
        elif(user_choice == "6"):
            show_rmse(house_df)
        elif(user_choice == "7"):
            print("Bye!")
        else:
            print("Bad input, try again!")
        user_choice = main_menu_choice()
    

def main_menu_choice():
    print("1. View General Summary")
    print("2. Show Price Prediction")
    print("3. Show OLS Summary")
    print("4. Chart Predicted Price")
    print("5. Show MAE")
    print("6. Show RMSE")
    print("7. Exit")

    return input("\nPlease choose an option: ")

def general_summary(house_df: pd.DataFrame):
    print(house_df[["sqft_living", "price", "grade", "floors", "age", "yr_built", "view"]].describe())

def show_price_prediction(house_df: pd.DataFrame):
    print(house_df[["price", "price_pred"]])

def show_ols_summary(ols_model):
    print(ols_model.summary())

def chart_pred_price(house_df: pd.DataFrame):
    plt.scatter(x=house_df["sqft_living"], y=house_df["price"], s=1, c="Blue", alpha=.4)
    plt.scatter(x=house_df["sqft_living"], y=house_df["price_pred"], s=1, c="Orange", alpha=.4)

    sqft_mean = house_df["sqft_living"].mean()
    price_mean = house_df["price"].mean()

    plt.axline((sqft_mean, price_mean), (sqft_mean * 1.5, price_mean * 1.5))

    plt.title("Sqft Living vs. House Price")
    plt.ylabel("House Price ($ millions)")
    plt.xlabel("Sqft Living")
    plt.show()

def show_mae(house_df: pd.DataFrame):
    mae = mean_absolute_error(house_df["price"], house_df["price_pred"])
    print(f"Mean Absolute Error: {mae}")

def show_rmse(house_df: pd.DataFrame):
    mse = mean_squared_error(house_df["price"], house_df["price_pred"])
    rmse = mse ** .5
    print(f"Root Mean Squared Error: {rmse}")

def read_house_data() -> pd.DataFrame:
    house_df = pd.read_csv(HOUSE_FILE, delimiter="\t")
    house_df["age"] = house_df["yr_built"].apply(lambda x: 2015 - x if pd.notna(x) else np.nan)
    return house_df



main()