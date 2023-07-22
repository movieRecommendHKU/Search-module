import re
import joblib
import numpy as np
from elasticsearch import Elasticsearch
from gensim.models import Word2Vec
from flask import Flask, request
from transformers import DistilBertTokenizer, DistilBertModel

app = Flask(__name__)

# Provide the path to the local directory where the model and tokenizer files are saved
model_path = "./ModelFiles/distilbert-base-uncased"
# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
# Load the model
model = DistilBertModel.from_pretrained(model_path)

pca = joblib.load("./ModelFiles/pca_model.pkl")
min_val = np.load('./ModelFiles/min_val.npy')
max_val = np.load('./ModelFiles/max_val.npy')

w2v = Word2Vec.load("./ModelFiles/word2vec.model")

es = Elasticsearch(hosts='http://localhost:9200')
# es = Elasticsearch(hosts='http://121.43.150.228:9200')

from keywordSearch import es_search_none, get_words_vector, es_search_keywords_and_vectors
from sentenceSearch import remove_punctuation, get_normalized_data, es_search_bert
@app.route('/SearchByKeywords', methods=['POST'])
def search_by_keywords():
    json_data = request.get_json()
    string_keywords = json_data["string_keywords"]
    k = json_data["k"]
    string_keywords = remove_punctuation(string_keywords)
    input_keywords = string_keywords.split()
    if len(input_keywords) == 0:
        search_movieId = es_search_none()
    else:
        input_vector = get_words_vector(input_keywords)
        search_movieId = es_search_keywords_and_vectors(string_keywords, input_vector, k)
    return search_movieId


@app.route('/SearchBySentences', methods=['POST'])
def search_by_sentences():
    json_data = request.get_json()
    string_sentence = json_data["string_sentence"]
    k = json_data["k"]
    input_sentence = re.split(r' |,|\.', string_sentence)
    input_sentence = list(filter(None, input_sentence))
    print(input_sentence)
    sentences_min_words = len(input_sentence)
    if sentences_min_words < 10:
        search_movieId = es_search_none()
    else:
        input_sentence = remove_punctuation(string_sentence)
        normalized_data_list = get_normalized_data(input_sentence)
        search_movieId = es_search_bert(normalized_data_list, k)
    return search_movieId


if __name__ == '__main__':
    app.run()
