import numpy as np
from app import w2v
from app import es

index_name = "movie_es_data"


def get_words_vector(input_keywords):
    input_vector = np.mean([w2v.wv[word] for word in input_keywords if word in w2v.wv], axis=0)
    return input_vector


def es_search_none():
    results_movieId = []
    return results_movieId


def es_search_keywords_and_vectors(input_words, input_vector, k):
    query = {
        "query": {
            "function_score": {
                "query": {
                    "multi_match": {
                        "query": input_words,
                        "fields": ["movieName", "overview", "directorName", "producerNames", "castNames", "keyWords", "genres"],
                        "fuzziness": "1"
                    }
                },
                "script_score": {
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'wordVectors') + 1.0",
                        "params": {"query_vector": input_vector.tolist()}
                    }
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
