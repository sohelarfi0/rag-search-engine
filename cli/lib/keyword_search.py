from lib.search_utils import load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
import os
import pickle

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
          self.index = defaultdict(set)
          self.docamp = {}
          self.index_path = CACHE_PATH / 'index.pkl'
          self.docamp_path = CACHE_PATH / 'docmap.pkl'
    def __add_document(self, doc_id, text):
        tokens =tokenize_text(text)
        for  token in set(tokens):
            self.index[token].add(doc_id)


        
            
    def get_documents(self, term):
        return sorted(list(self.index[term]))
    

    
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docamp[doc_id] = movie

        

    def save(self):
        os.mkdirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index,f)
        with open(self.docamp_path, 'wb') as f:
            pickle.dump(self.docamp, f)


def build_command():
    idx=InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' ={docs[0]}")

        



def clean_text(text):
    text= text.lower()
    text= text.translate(str.maketrans("", "" , string.punctuation))
    return text

def tokenize_text(text):
    text = clean_text(text)
    stopwords = load_stopwords()
    res = []
    def _filter(tok):
        if tok  and tok not in stopwords:
            return True
        return False
    for tok in text.split():
        if _filter(tok):
            tok = stemmer.stem(tok)
            res.append(tok)
    return res

def has_matching_token(query_tokens, movie_tokens):
    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
            
    return False        

def search_command(query, n_results):
    movies=load_movies()
    res=[]
    query_tokens = tokenize_text(query)
    for movie in movies:
        movie_tokens =tokenize_text(movie['title'])
        if has_matching_token(query_tokens,movie_tokens):
            res.append(movie)

        if len(res) == n_results:
            break
    return res
        
