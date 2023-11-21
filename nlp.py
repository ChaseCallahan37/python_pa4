import pandas as pd
import numpy as np
import gzip
from dateutil.parser import parse as date_parser
from textblob import TextBlob
import re
import os.path as path
from calendar import month_name
import matplotlib.pyplot as plt


REVIEWS_FILE_ORIGINAL = "reviews_Office_Products_5.json.gz"
REVIEWS_FILE_CSV = "review_data.csv"
REVIEW_FILE_CLEANED = "review_data_cleaned.csv"

def main():
    reviews_df = get_reviews_df()
    print(reviews_df.describe())
    print(reviews_df[["overall", "wordCount", "sentimentScore"]].describe())

    monthly_reviews_df = pd.DataFrame({"month" : list(map(lambda x: x, range(1, 13)))})
    monthly_reviews_df.reset_index().set_index(["month"])

    reviews_by_month_df = reviews_df.groupby('month').size().to_frame('count')
    
    monthly_reviews_df["count"] = monthly_reviews_df["month"].apply(lambda x: reviews_by_month_df.loc[x]["count"])
    monthly_reviews_df = monthly_reviews_df.set_index("month")
    print(monthly_reviews_df)
    
    monthly_reviews_df["monthName"] = [month_name[i] for i in monthly_reviews_df.index]
    
    # plt.bar(x=monthly_reviews_df["monthName"], height=monthly_reviews_df["count"])
    # plt.plot(monthly_reviews_df["monthName"], monthly_reviews_df["count"], color="red")
    # plt.xticks(rotation=45)
    # plt.title("Number of Reviews by Mont")
    # plt.ylabel("Number of Reviews")
    # plt.xlabel("Month")
    # plt.show()

    print(reviews_df[["overall", "sentimentType"]])
    sentiment_df = reviews_df.groupby(["overall", "sentimentType"]).size().to_frame("count").reset_index()
    sentiment_pivot = sentiment_df.pivot_table(index=["overall"], columns=["sentimentType"], values=["count"])

    # sentiment_pivot = pd.pivot_table(index=sentiment_df["overall"], columns=sentiment_df["sentimentType"], data=sentiment_df["count"])
    print(sentiment_pivot)

def get_reviews_df() -> pd.DataFrame:
    if(path.isfile(REVIEW_FILE_CLEANED)):
       return pd.read_csv(REVIEW_FILE_CLEANED)
    
    reviews_df = getDF()
    reviews_df["reviewTime"] = reviews_df["reviewTime"].apply(date_parser)
    reviews_df["month"] = reviews_df["reviewTime"].apply(lambda x: x.month)
    reviews_df["year"] = reviews_df["reviewTime"].apply(lambda x: x.year)
    reviews_df["reviewText"] = reviews_df["reviewText"].apply(clean_review_text)
    reviews_df["wordCount"] = reviews_df["reviewText"].apply(get_word_count)
    reviews_df["sentimentScore"] = reviews_df["reviewText"].apply(lambda x: TextBlob(x).sentiment[0] if pd.notna(x) else np.nan)
    reviews_df["sentimentType"] = reviews_df["sentimentScore"].apply(get_sentiment_type)

    reviews_df.to_csv(REVIEW_FILE_CLEANED)

    return reviews_df

def clean_review_text(review):
   if(pd.isna(review)):
      return np.nan
   review = re.sub("[^a-zA-Z ]", "", review)
   review = re.sub("[ ]{2,}", " ", review)
   return review.strip()

def get_word_count(text):
   if(pd.isna(text) or text == ""):
      return 0
   return (len(text.split(" ")) + 1)

def get_sentiment_type(polarity):
   if(pd.isna(polarity)):
      return np.nan
   if(polarity > 0):
      return "positive"
   if(polarity == 0):
      return "neutral"
   if(polarity < 0):
      return "negative"

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF() -> pd.DataFrame:
  if(path.isfile(REVIEWS_FILE_CSV)):
     return pd.read_csv(REVIEWS_FILE_CSV)
  i = 0
  df = {}
  for d in parse(REVIEWS_FILE_ORIGINAL):
    df[i] = d
    i += 1
  df = pd.DataFrame.from_dict(df, orient='index')
  df.to_csv(REVIEWS_FILE_CSV)
  return df



main()