import pandas as pd
import sys; sys.path.append('../')
PATH_RATINGS = f"../data/ratings.csv"
PATH_NETWORK = f"../data/beer_network.csv"
PATH_BEER_META = f"../data/beers_metadata.csv"

def load_data():
    df_network = pd.read_csv(PATH_NETWORK)
    df_beer_meta = pd.read_csv(PATH_BEER_META)
    df_user_ratings = pd.read_csv(PATH_RATINGS)
    return df_network, df_beer_meta, df_user_ratings