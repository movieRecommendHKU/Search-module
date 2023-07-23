from app import es

index_name = "user_es_data"


def es_search_userSimilarity(input_vector, k):
    query = {
        "query": {
            "function_score": {
                "script_score": {
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'similarity') + 1.0",
                        "params": {"query_vector": input_vector}
                    }
                }
            }
        },
        "size": k
    }
    results_all = es.search(index=index_name, body=query)
    results_userId = []
    if results_all['hits']['total']['value'] > 0:
        for hit in results_all['hits']['hits']:
            user_id = hit['_source']['userId']
            similarity = hit['_score']
            results_userId.append(user_id)
            print(f"User ID: {user_id} - Similarity: {similarity}")
    else:
        print("No user found matching the input keywords.")
    return results_userId
