#!/usr/bin/env python3

import argparse
from lib.keyword_search import (search_command , build_command, 
                                tf_command, idf_command, tfidf_command, 
                                bm25_idf_command, bm25_tf_command,bm25_search);
from lib.search_utils import BM25_K1; 


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
   
    search_parser = subparsers.add_parser("build", help="Build Cache ")
    # search_parser.add_argument("query", type=str, help="Search query")
    
    search_parser = subparsers.add_parser("tf", help="Search movies using BM25")
    search_parser.add_argument("doc_id", type=str, help="Document ID for to check")
    search_parser.add_argument("term", type=str, help="Search  term to find counts for")
    
    search_parser = subparsers.add_parser("idf", help="Calculate Inverted  document frequency ")
    # search_parser.add_argument("doc_id", type=str, help="Document ID for to check")
    search_parser.add_argument("term", type=str, help="Search  term to find counts for")
   
    search_parser = subparsers.add_parser("tfidf", help="Calculate term frequency- Inverted  document frequency ")
    search_parser.add_argument("doc_id", type=str, help="Document ID for to check")
    search_parser.add_argument("term", type=str, help="Search  term to find counts for")
    
    bm25_idf_parser = subparsers.add_parser(
        'bm25idf', help ="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    
    bm25_tf_parser = subparsers.add_parser(
        'bm25tf', help ="Get BM25 TF score for a given term"
    )
    bm25_tf_parser.add_argument("doc_id", type=str, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("term", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 TF ")
    
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full title")
    bm25search_parser.add_argument("query", type=str, help="Search query")
 
    args = parser.parse_args()

    match args.command:

        case "bm25search":
            bm25_results = bm25_search(args.query)
            for idx, res in enumerate(bm25_results):
                print(f"{idx}. {res['doc_id']} {res['title']} - Search {res['score']}")


        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}':{bm25tf:.2f}")
        case "search":
            print(f"Searching for: {args.query}")
            results= search_command(args.query , 5)
            for i, result in enumerate(results):
                print(f"{i} {result['title']}")
        case "build":
            build_command()  

        case "tf":
            tf_command(args.doc_id , args.term)  

        case "idf":
            idf_command(args.term)  
        case  "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM@% IDF score of '{args.term}' : {bm25idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()