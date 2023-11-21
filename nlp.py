import pandas as pd
import gzip

REVIEWS_FILE = "reviews_Office_Products_5.json.gz"

def main():
    reviews_df = getDF(REVIEWS_FILE)
    print(reviews_df)

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')



main()