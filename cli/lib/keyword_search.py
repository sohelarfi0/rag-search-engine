from lib.search_utils import (BM25_K1,BM25_B, load_movies, load_stopwords ,CACHE_PATH, CACHE_DIR, Counter)
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
import os
import pickle
import math

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
          self.index = defaultdict(set)
          self.term_frequencies = defaultdict(Counter)
          self.index_path= CACHE_DIR/'index.pkl'
          self.docamp = {}


        #   self.index_path = CACHE_PATH / 'index.pkl'
          self.docamp_path = CACHE_DIR / 'docmap.pkl'
          self.term_frequencies_path = CACHE_DIR/ 'term_frequencies.pkl'
          self.doc_lengths = {}
          self.doc_lengths_path =CACHE_DIR/'term_frequencies.pkl'

    def __add_document(self, doc_id, text):
        tokens =tokenize_text(text)
        for  token in set(tokens):
            self.index[token].add(doc_id).add(doc_id)
        self.term_frequencies[doc_id].update(tokens)   
        self.doc_lengths[doc_id] = len(tokens)


    def _get_avg_doc_length(self):
        lengths = self.doc_lengths.values()
        if len(lengths) == 0:
            return 0.0
        ttl = 0
        for l in lengths:
            ttl += l
            return ttl / len(lengths)
        

        
            
    def get_documents(self, term):
        return sorted(list(self.index[term]))
    
    def bm25_tf_command(doc_id, term, k1=BM25_K1):
        inverted_index = InvertedIndex()
        inverted_index.load()
        return inverted_index.get_bm25_tf(term, k1)
    
    
    def bm25_idf_command(term):
        inverted_index = InvertedIndex()
        inverted_index.load()
        return inverted_index.get_bm25_idf(term)
    
    
    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token)!= 1:
            raise ValueError("Can only have 1 tokens")
        return self.term_frequencies[doc_id][token[0]]
    
    def get_bm25_tf(self, doc_id, term, k1 = BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths(doc_id)
        avg_doc_length = self._get_avg_doc_length[doc_id]
        if avg_doc_length>0:
            length_norm = 1

        return (tf * (k1 + 1))/(tf+k1 * length_norm)
    


    def get_idf(self, term):
        token= tokenize_text(term)
        if len(token)!= 1:
            raise ValueError("Can only have 1 tokens")
        token = token[0]
        doc_count = len(self.docamp)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1)/(term_doc_count + 1 ))
        
    def get_bm25_idf(self, term:str)->float:
        token= tokenize_text(term)
        if len(token)!= 1:
            raise ValueError("Can only have 1 tokens")
        token = token[0]
        doc_count = len(self.docamp)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5)/(term_doc_count+0.5)+1)

     

    def bm25_search(query):
        inverted_index = InvertedIndex()
        inverted_index.load()
        return inverted_index.bm25_search(query)

    def get_tfidf(self,doc_id,term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf*idf
     
    def get_bm25(self,doc_id,term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf*idf
    
    def bm25_search(self, query, limit):
        query_tokens = tokenize_text(query)
        scores={}
        for doc_id in self.docamp:
            score = 0
            for token in query_tokens:
                score += self.get_bm25(doc_id, token)
            scores[doc_id] = score 

            results = sorted_scores[: limit]
            format_rsults = []
            for doc_id, score in results:
                title = self.docamp[doc_id]['title']
                format_rsults.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "score": score
                    }
                )

            return format_rsults    


        def _key(x): return x[1]
        sorted_scores = sorted(scores.items(),
                               key=lambda x: x[1],
                               reverse= True)
    

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
        with open(self.term_frequencies_path,"wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)



    def load(self):
        with open(self.index_path, "rb") as f:
         self.index = pickle.load(f)
        with open(self.docamp_path,"rb") as f:
            self.docamp = pickle.load(f)  
        with open(self.term_frequencies_path ,"rb") as f:
            self.docamp = pickle.load(f)

        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)    

def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tfidf = idx.get_tfidf(doc_id, term)
    print(f"TF_IDF score of '{term}' in document '{doc_id}' : {tfidf:.2f}")


def idf_command(term):
    idx= InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    print(f"Inverse document frequency pf '{term}' : {idf:2f}")
                

               
def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


    # docs = idx.get_documents("merida")
    # print(f"First document for token 'merida' ={docs[0]}")

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

def search_command(query, n_results=5):
    movies=load_movies()
    idx = InvertedIndex()
    idx.load()
    seen, res=set(), []
    query_tokens = tokenize_text(query)
    # for movie in movies:
    #     movie_tokens =tokenize_text(movie['title'])
    #     if has_matching_token(query_tokens,movie_tokens):
    #         res.append(movie)

    #     if len(res) == n_results:
    #         break


    for qt in query_tokens:
        matching_doc_ids =idx.get_documents(qt)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen :
                continue
            seen.add(matching_doc_id)
            matching_doc = idx.docamp[matching_doc_id]
            res.append(matching_doc
                       )
            
            if len(res) >= n_results:
             return res
        
