import json
from pathlib import Path
from collections import Counter

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'
MOVIES_PATH = DATA_PATH / 'movie.json'
STOPWORDS_PATH = DATA_PATH / 'stopwords.txt'

CACHE_PATH = DATA_PATH / 'cache'
CACHE_DIR = CACHE_PATH



def load_movies() ->list[dict]:
    with open(DATA_PATH, "r") as f:
        data=json.load(f)

    return data['movies']  

def load_stopwords():
    with open(DATA_PATH,"r") as f:
        data = f.readlines()
    return data  
