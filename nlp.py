import pandas as pd
import numpy as np
import gzip
from dateutil.parser import parse as date_parser
from textblob import TextBlob
import re
import os.path as path


REVIEWS_FILE_ORIGINAL = "reviews_Office_Products_5.json.gz"
REVIEWS_FILE_CSV = "review_data.csv"
REVIEW_FILE_CLEANED = "review_data_cleaned.csv"

def main():
    reviews_df = get_reviews_df()
    print(reviews_df.describe())

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
      return 1
   if(polarity == 0):
      return 0
   if(polarity < 0):
      return -1

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