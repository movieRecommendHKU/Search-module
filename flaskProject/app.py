import re
from flask import Flask
from keywordSearch import get_words_vector, es_search_none, es_search_keywords_and_vectors
from sentenceSearch import remove_punctuation, get_normalized_data, es_search_bert

app = Flask(__name__)

index_name = "movie_es_data"


@app.route('/SearchByKeywords', methods=['POST'])
def search_by_keywords(string_keywords, k):
    input_keywords = re.split(' |,|.', string_keywords)
    if len(input_keywords) == 0:
        search_movieId = es_search_none()
    else:
        input_words, input_vector = get_words_vector(input_keywords)
        search_movieId = es_search_keywords_and_vectors(input_words, input_vector, k)
    return search_movieId


@app.route('/SearchBySentences', methods=['POST'])
def search_by_sentences(string_sentence, k):
    sentences_min_words = len(re.split(' |,|.', string_sentence))
    if sentences_min_words < 10:
        search_movieId = es_search_none()
    else:
        input_sentence = remove_punctuation(string_sentence)
        normalized_data_list = get_normalized_data(input_sentence)
        search_movieId = es_search_bert(normalized_data_list, k)
    return search_movieId


if __name__ == '__main__':
    app.run(port=8098)
