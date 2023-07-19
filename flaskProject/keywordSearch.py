import numpy as np
from Models.word2vecModel import w2v
from esClient import es
from app import index_name


def get_words_vector(input_keywords):
    input_words = [word for phrase in input_keywords for word in phrase.split()]
    input_vector = np.mean([w2v.wv[word] for word in input_words if word in w2v.wv], axis=0)
    return input_words, input_vector


def es_search_none():
    results_movieId = []
    return results_movieId


def es_search_keywords_and_vectors(input_words, input_vector, k):
    query = {
        "query": {
            "script_score": {
                "query": {"match_all": {
                    "query": input_words,
                    "fuzziness": "1"
                }},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'wordVectors') + 1.0",
                    "params": {"query_vector": input_vector.tolist()}
                }
            }
        },
        "size": k
    }
    results_all = es.search(index=index_name, body=query)
    results_movieId = []
    if results_all['hits']['total']['value'] > 0:
        for hit in results_all['hits']['hits']:
            movie_id = hit['_source']['movieId']
            similarity = hit['_score']
            results_movieId.append(movie_id)
            print(f"Movie ID: {movie_id} - Similarity: {similarity}")
    else:
        print("No movies found matching the input keywords.")
    return results_movieId
