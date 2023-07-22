import re
import numpy as np
import torch
import torch.nn.functional as F
from app import tokenizer,model
from app import pca, min_val, max_val
from sklearn.preprocessing import MinMaxScaler
from app import es

max_length = 40 # bert 句子向量的最大值
input_length = 23 # torch.Size([43486, 23, 768]) 这个要跟中间的23长度对应（第四个代码框），因为description里面长度影响结果
n_components = 900 #PCA降维后的维度
movie_num = 43449 # movie number
index_name = "movie_es_data"
# input_sentence means the description user input
# host is:'http://localhost:9200'

# 需要保存三个文件 第一个是pca矩阵 第二个是normalized的最大值最小值数组

def remove_punctuation(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)  # Remove Chinese garbled characters
    return cleaned_text

def get_normalized_data(input_sentence):
    input_tokens = tokenizer.encode_plus(
        input_sentence,
        truncation=True,
        max_length=input_length - 2,  # Consider adding the length of the start and end tags
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = input_tokens['input_ids']  # Enter the encoding of the sentence

    # Add start and end tags to the beginning and end of the input sentence
    if len(input_ids) < input_length:
        input_ids = torch.cat(
            [torch.tensor([tokenizer.cls_token_id]), input_ids.squeeze(), torch.tensor([tokenizer.sep_token_id])])
    #     attention_mask = torch.cat([torch.tensor([1]), attention_mask.squeeze(), torch.tensor([1])])

    # If the length of the input sentence is less than input length, fill it in
    if len(input_ids) < input_length:
        num_pad_tokens = input_length - len(input_ids)
        input_ids = F.pad(input_ids, (0, num_pad_tokens), value=tokenizer.pad_token_id)
    #     attention_mask = F.pad(attention_mask, (0, num_pad_tokens), value=0)

    # Converts the input to a tensor and gets an embedded representation
    input_ids = input_ids.unsqueeze(0)
    # attention_mask = attention_mask.unsqueeze(0)

    with torch.no_grad():
        input_outputs = model(input_ids)
        input_embedding = input_outputs.last_hidden_state.squeeze(0)

    embedding_np = input_embedding.detach().numpy().reshape(-1).tolist()

    # Convert embedding_np to a NumPy array
    embedding_np_array = np.array(embedding_np)

    # The transformation was performed using the previously trained PCA model
    input_pca = pca.transform(embedding_np_array.reshape(1, -1))

    # Use MinMaxScaler for standardization
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    scaler2.data_min_ = min_val
    scaler2.data_max_ = max_val

    normalized_data = (input_pca - min_val) / (max_val - min_val)

    # Convert the result to a 1x900 list
    normalized_data_list = np.squeeze(normalized_data).tolist()

    return normalized_data_list

def es_search_bert(query_vector, k):
    query = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'sentenceVectors') + 1.0",
                    "params": {"query_vector": query_vector}
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
            movie_name = hit['_source']['movieName']
            similarity = hit['_score']
            results_movieId.append(movie_id)
            print(f"Movie ID: {movie_id} - Movie Name: {movie_name} - Similarity: {similarity}")
    else:
        print("No movies found matching the input keywords.")
    return results_movieId
